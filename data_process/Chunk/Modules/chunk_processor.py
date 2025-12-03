# Modules/chunk_processor.py
import copy
import re
import os
import shutil
from .logger import get_chunk_logger

def clear_chunks_directory():
    path = "chunks"
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"已清除舊的 {path}/ 目錄")
    else:
        print("chunks/ 不存在，無需清除")

def process_json_chunks(data, max_chars=1000, overlap_ratio=0.2):
    logger = get_chunk_logger()
    logger.log_start()
    logger.log_config(max_chars, overlap_ratio)

    chunks = []
    prev_record = None

    for item in data["content"]:
        key = (item["heading"], item["subheading"], item["title"])
        logger.info(f"正在處理章節: {key}")

        paragraphs = item.get("text") or [""]
        i = 0  # 段落指針

        while i < len(paragraphs):
            current_paras = []
            current_chars = 0

            # === 1. 嘗試插入前一塊的重疊（只有同一個 key 才插）===
            overlap_inserted = False
            if (prev_record
                and prev_record["key"] == key
                and prev_record["overlap"]):
                overlap_chars = sum(len(p) for p in prev_record["overlap"])
                # 重疊不能超過 max_chars 的 50%，否則會卡死
                if overlap_chars <= max_chars * 0.5:
                    current_paras = prev_record["overlap"][:]
                    current_chars = overlap_chars
                    overlap_inserted = True
                    logger.info(f"Chunk {len(chunks)+1}: 成功插入重疊 {overlap_chars} 字元")

            # === 2. 貪婪加入新段落，至少要加入一條新段落 ===
            added_new = False
            while i < len(paragraphs):
                next_para = paragraphs[i]
                if current_chars + len(next_para) <= max_chars:
                    current_paras.append(next_para)
                    current_chars += len(next_para)
                    i += 1
                    added_new = True
                else:
                    break

            # 關鍵防呆：如果因為重疊太大導致完全沒加新段落，強制加一段並丟棄過大的重疊
            if not added_new and i < len(paragraphs):
                forced_para = paragraphs[i]
                current_paras.append(forced_para)
                i += 1
                logger.info(f"Chunk {len(chunks)+1}: 重疊過大，已強制加入新段落（{len(forced_para)} 字元）並丟棄舊重疊")

            # === 3. 產生 chunk ===
            if current_paras:
                chunk = copy.deepcopy(data)
                chunk["content"] = [{
                    "heading": item["heading"],
                    "subheading": item["subheading"],
                    "title": item["title"],
                    "text": current_paras
                }]
                chunks.append(chunk)

                logger.log_chunk_content(len(chunks), "".join(current_paras))

                # === 4. 計算下一塊要用的重疊段落 ===
                full_text = "".join(current_paras)
                target_overlap_chars = max(100, int(len(full_text) * overlap_ratio))  # 至少100字，防止太小

                overlap_paras = []
                accumulated = 0
                for p in reversed(current_paras):
                    if accumulated + len(p) <= target_overlap_chars:
                        overlap_paras.insert(0, p)
                        accumulated += len(p)
                    else:
                        if not overlap_paras:  # 至少留一段
                            overlap_paras.insert(0, p)
                        break

                prev_record = {
                    "key": key,
                    "overlap": overlap_paras
                }

            else:
                # 理論上不會走到這裡，但保險起見直接跳過
                i += 1

    logger.log_completion(len(chunks))
    return chunks