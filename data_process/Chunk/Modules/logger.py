# Modules/logger.py
import os
import logging
from datetime import datetime

_logger = None

def get_chunk_logger(enable_logging=True, log_file_path="log/ChunkLog.log"):
    """
    完整的 Chunk Logger（支援開關、自訂路徑、所有需要的 log 方法）
    """
    global _logger
    if _logger is not None:
        return _logger

    logger = logging.getLogger("ChunkLogger")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    if enable_logging:
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        fh = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(asctime)s | %(message)s', '%H:%M:%S'))
        logger.addHandler(fh)
        print(f"日誌將寫入 → {os.path.abspath(log_file_path)}")
    else:
        logger.addHandler(logging.NullHandler())

    # === 以下是所有會被用到的方法 ===
    def log_start(self):
        self.info("=== Chunk 處理開始 ===")
        self.info(f"開始時間: {datetime.now():%Y-%m-%d %H:%M:%S}")

    def log_config(self, max_chars, overlap_ratio):
        self.info(f"分塊設定: max_chars={max_chars}, overlap_ratio={overlap_ratio}")

    def log_chunk_content(self, chunk_id, content):
        self.info(f"--- Chunk {chunk_id} 內容開始 ---")
        # 避免單行太長，直接寫原始文字（最多顯示前500字）
        preview = content.replace('\n', '⏎ ')[:500]
        self.info(f"內容預覽: {preview}")
        if len(content) > 500:
            self.info("...（內容過長，省略後續）")
        self.info(f"字元數: {len(content)}")
        self.info(f"--- Chunk {chunk_id} 內容結束 ---\n")

    def log_completion(self, count):
        self.info(f"分塊完成！共產生 {count} 個 chunk")
        self.info(f"結束時間: {datetime.now():%Y-%m-%d %H:%M:%S}")
        self.info("=" * 50 + "\n")

    # 綁定到 logger 實例
    logger.log_start = log_start.__get__(logger)
    logger.log_config = log_config.__get__(logger)
    logger.log_chunk_content = log_chunk_content.__get__(logger)
    logger.log_completion = log_completion.__get__(logger)

    _logger = logger
    return logger