#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
关系提取器模块
负责从文本中提取实体间的关系
"""

import logging
import json
import re
from pathlib import Path
from typing import Union, Dict, Any, List
from LLM import LLM


class RelationExtractor:
    """
    完全参数化的关系提取器。
    所有关键配置（模型、prompt、输入/输出路径）均由调用方显式传入，
    模块内部不做任何路径或命名假设。
    """

    def __init__(self, model_name='default', threshold=0.5):
        """
        初始化关系提取器

        Args:
            model_name (str): 使用的模型名称
            threshold (float): 关系识别的置信度阈值
        """
        self.model_name = model_name
        self.threshold = threshold
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"初始化关系提取器，模型: {model_name}，阈值: {threshold}")

    def extract_single(self, record: Dict, prompt_path: str) -> Dict:
        """
        处理单条记录，提取关系。

        Args:
            record (dict): 单条输入记录（应包含 "results" 字段，即实体信息）
            prompt_path (str): Prompt 模板文件路径

        Returns:
            dict: 更新后的记录，包含 "results" 字段（关系列表）

        Raises:
            Exception: 任何处理失败都会抛出异常（由调用方捕获）
        """
        if not isinstance(record, dict):
            raise ValueError("输入 record 必须是字典类型")

        # 加载 prompt（简单可靠；若性能敏感可外部缓存后传入字符串）
        with open(prompt_path, 'r', encoding='utf-8') as f:
            system_prompt = f.read().strip()

        user_content = json.dumps(record, ensure_ascii=False, indent=2)
        full_prompt = f"{system_prompt}\n\n{user_content}"

        llm = LLM(prompt=full_prompt, api_provider=self.model_name)
        raw_response = llm.llm_call()
        response_text = llm.extract_response(raw_response)

        cleaned = self._clean_text(response_text)
        try:
            parsed = json.loads(cleaned)
            if not isinstance(parsed, dict) or "results" not in parsed:
                raise ValueError("LLM 返回结果缺少 'results' 字段")
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.warning(f"LLM 返回无效 JSON | 片段: {cleaned[:150]}...")
            raise ValueError(f"无法解析 LLM 输出为有效关系结果: {e}")

        # 构建最终记录：保留原字段，仅更新 results
        final = {k: v for k, v in record.items() if k != "results"}
        final["results"] = parsed.get("results", [])
        return final

    def extract(
            self,
            input_file: Union[str, Path],
            output_file: Union[str, Path],
            prompt_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """
        【兼容旧接口】执行整文件关系提取（调用 extract_single）

        Args:
            input_file: 输入 JSONL 文件路径（必须存在）
            output_file: 输出 JSONL 文件路径
            prompt_path: Prompt 文件路径（必须存在）

        Returns:
            dict: 包含状态和统计信息的结果
        """
        input_file = Path(input_file)
        output_file = Path(output_file)
        prompt_path = Path(prompt_path)

        if not input_file.exists():
            raise FileNotFoundError(f"输入文件不存在: {input_file}")
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt 文件不存在: {prompt_path}")

        output_file.parent.mkdir(parents=True, exist_ok=True)

        records = self._load_jsonl(str(input_file))
        total = len(records)
        success_count = 0

        with open(output_file, 'w', encoding='utf-8') as out_f:
            for i, record in enumerate(records, 1):
                try:
                    result = self.extract_single(record, str(prompt_path))
                    out_f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    success_count += 1
                except Exception as e:
                    self.logger.error(f"处理第 {i} 条记录失败: {e}", exc_info=True)
                    # 保持旧逻辑：失败时输出空 relations（不中断流程）
                    fallback = self._build_fallback_record(record)
                    out_f.write(json.dumps(fallback, ensure_ascii=False) + '\n')

        return {
            "status": "completed",
            "output_path": str(output_file.resolve()),
            "total_records": total,
            "successful_records": success_count
        }

    def _build_fallback_record(self, record: Dict) -> Dict:
        """构建失败时的回退记录（保留原 entities，relations 置空）"""
        fallback = {k: v for k, v in record.items() if k != "results"}
        results = []
        for res in record.get("results", []):
            if isinstance(res, dict) and "output" in res:
                output = res["output"]
                if isinstance(output, dict):
                    output = {**output, "relations": []}
                res = {**res, "output": output}
            results.append(res)
        fallback["results"] = results
        return fallback

    def _load_jsonl(self, path: str) -> List[Dict]:
        records = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"跳过无效行 {line_num}: {e}")
        return records

    def _clean_text(self, text: str) -> str:
        return re.sub(r'[\x00-\x1f\x7f-\x9f]', '', str(text or ""))

    # --- 预留接口（保持不变） ---
    def load_model(self):
        self.logger.info(f"加载关系提取模型: {self.model_name}")
        return None

    def preprocess(self, text, entities):
        return text, entities

    def postprocess(self, predictions):
        return []
