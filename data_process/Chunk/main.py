# main.py
import json
import os
import glob
from Modules.chunk_processor import clear_chunks_directory, process_json_chunks
from Modules.logger import get_chunk_logger

def main():
    try:
        # ====================== 所有設定都在這裡 ======================
        max_chars         = 400
        overlap_ratio     = 0.2

        input_path        = "../../../Input/inputData.json"
        output_chunks_dir = "../../../Input/chunks"

        enable_logging    = True          # 改成 True 就會寫 log
        log_file_path     = "../../../Output/Log/chunksLog.log"
        # =============================================================

        print("=== 開始全新運行 ===")
        print(f"分塊參數 → max_chars={max_chars}, overlap_ratio={overlap_ratio}")
        print(f"輸入檔案 → {os.path.abspath(input_path)}")
        print(f"輸出目錄 → {os.path.abspath(output_chunks_dir)}")
        print(f"日誌開關 → {'開啟' if enable_logging else '關閉'}")

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"找不到輸入檔案：{input_path}")

        # ← 這一行一定要能接受兩個參數
        logger = get_chunk_logger(enable_logging=enable_logging, log_file_path=log_file_path)
        logger.log_start()

        clear_chunks_directory()

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print("讀取 inputData.json 完成")

        print("開始純 Python 分塊處理...")
        chunks = process_json_chunks(data, max_chars=max_chars, overlap_ratio=overlap_ratio)

        # 手動寫檔（路徑完全由 main 控制）
        os.makedirs(output_chunks_dir, exist_ok=True)
        for f in glob.glob(os.path.join(output_chunks_dir, "chunk_*.json")):
            os.remove(f)

        for i, chunk in enumerate(chunks, 1):
            filepath = os.path.join(output_chunks_dir, f"chunk_{i:03d}.json")
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(chunk, f, ensure_ascii=False, indent=2)

        print(f"\n=== 運行成功 ===")
        print(f"共產生 {len(chunks)} 個 chunk")
        print(f"已全部儲存至 → {output_chunks_dir}")
        if enable_logging:
            print(f"詳細日誌 → {os.path.abspath(log_file_path)}")

        logger.log_completion(len(chunks))

    except Exception as e:
        print(f"錯誤: {e}")
        raise

if __name__ == "__main__":
    main()