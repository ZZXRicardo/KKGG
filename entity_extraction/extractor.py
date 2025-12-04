#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
实体提取器模块
负责从文本中提取实体
"""

import logging
import json
from pathlib import Path
from LLM import LLM


class EntityExtractor:

    def __init__(self, model_name='default', threshold=0.5):
        """
        初始化实体提取器

        Args:
            model_name (str): 使用的模型名称（如 'qwen', 'deepseek'）
            threshold (float): 实体识别的置信度阈值（当前未使用，保留扩展性）
        """
        self.model_name = model_name
        self.threshold = threshold
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"初始化实体提取器，模型: {model_name}，阈值: {threshold}")

    def extract_single(self, record: dict, prompt_path: str) -> dict:
        """
        处理单条记录，返回包含实体的结果字典。

        Args:
            record (dict): 单条输入记录（原始 JSON 对象）
            prompt_path (str): Prompt 模板文件路径

        Returns:
            dict: 更新后的记录，包含 "results" 字段（实体列表）

        Raises:
            Exception: 任何处理失败都会抛出异常（由调用方捕获）
        """
        if not isinstance(record, dict):
            raise ValueError("输入 record 必须是字典类型")

        # 加载 prompt（每次调用都读取，简单可靠；若性能敏感可缓存）
        with open(prompt_path, 'r', encoding='utf-8') as f:
            system_prompt = f.read().strip()

        user_input = json.dumps(record, ensure_ascii=False, indent=2)
        full_prompt = f"{system_prompt}\n\n---\n\n{user_input}"

        llm = LLM(prompt=full_prompt, api_provider=self.model_name)
        raw_response = llm.llm_call()
        extracted_text = llm.extract_response(raw_response)

        try:
            result_obj = json.loads(extracted_text)
            entities = result_obj.get("results", [])
        except json.JSONDecodeError as e:
            self.logger.warning(f"LLM 返回非 JSON 格式: {extracted_text[:100]}...")
            entities = []

        # 返回新对象，不修改原 record
        final_record = record.copy()
        final_record["results"] = entities
        return final_record

    def extract(self, input_file, output_file, prompt_path):
        """
        【兼容旧接口】执行整文件实体提取任务（调用 extract_single）

        Args:
            input_file (str or Path): 输入 JSON 文件路径（必须是包含对象列表的 JSON 数组）
            output_file (str or Path): 输出 JSONL 文件路径
            prompt_path (str or Path): Prompt 文件路径

        Returns:
            dict: 包含状态和输出路径的结果字典
        """
        input_file = Path(input_file)
        output_file = Path(output_file)
        prompt_path = Path(prompt_path)

        self.logger.info(f"开始从 {input_file} 提取实体，使用模型: {self.model_name}, Prompt: {prompt_path}")

        if not input_file.is_file():
            raise FileNotFoundError(f"输入文件不存在: {input_file}")
        if not prompt_path.is_file():
            raise FileNotFoundError(f"Prompt 文件不存在: {prompt_path}")

        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"输入 JSON 文件的根元素必须是一个数组（list），但得到: {type(data).__name__}")

        processed = 0
        with open(output_file, 'w', encoding='utf-8') as out_f:
            for idx, record in enumerate(data):
                try:
                    result = self.extract_single(record, str(prompt_path))
                    out_f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    processed += 1
                except Exception as e:
                    self.logger.error(f"处理记录索引 {idx} 时出错: {e}", exc_info=True)
                    # 旧逻辑：失败条目仍输出空 results
                    fallback = record.copy()
                    fallback["results"] = []
                    out_f.write(json.dumps(fallback, ensure_ascii=False) + '\n')
                    processed += 1

        self.logger.info(f"实体提取完成，共处理 {processed} 条记录，结果保存至 {output_file}")
        return {
            "status": "success",
            "output_path": str(output_file.resolve()),
            "processed": processed
        }

    def load_model(self):
        """预留接口"""
        self.logger.info(f"加载实体提取模型: {self.model_name}")
        return None

    def preprocess(self, text):
        """预留接口"""
        return text

    def postprocess(self, predictions):
        """预留接口"""
        return []
