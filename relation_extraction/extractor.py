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

    def extract(
        self,
        input_file: Union[str, Path],
        output_file: Union[str, Path],
        prompt_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """
        执行关系提取
        
        Args:
            input_file: 输入 JSONL 文件路径（必须存在）
            output_file: 输出 JSONL 文件路径（父目录将自动创建）
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

        with open(prompt_path, encoding='utf-8') as f:
            system_prompt = f.read().strip()

        records = self._load_jsonl(str(input_file))
        total = len(records)
        success_count = 0

        with open(output_file, 'w', encoding='utf-8') as out_f:
            for i, record in enumerate(records, 1):
                try:
                    updated = self._process_single_record(record, system_prompt)
                    out_f.write(json.dumps(updated, ensure_ascii=False) + '\n')
                    success_count += 1
                except Exception as e:
                    self.logger.error(f"处理第 {i} 条记录失败: {e}", exc_info=True)
                    fallback = self._build_fallback_record(record)
                    out_f.write(json.dumps(fallback, ensure_ascii=False) + '\n')

        output_path = str(output_file.resolve())
        return {
            "status": "completed",
            "output_path": output_path,
            "total_records": total,
            "successful_records": success_count
        }

    def _process_single_record(self, record: Dict, system_prompt: str) -> Dict:
        user_content = json.dumps(record, ensure_ascii=False, indent=2)
        full_prompt = f"{system_prompt}\n\n{user_content}"

        llm = LLM(prompt=full_prompt, api_provider=self.model_name)
        raw_response = llm.llm_call()
        response_text = llm.extract_response(raw_response)

        cleaned = self._clean_text(response_text)
        try:
            parsed = json.loads(cleaned)
            if not isinstance(parsed, dict) or "results" not in parsed:
                raise ValueError("缺少 'results' 字段")
        except (json.JSONDecodeError, ValueError):
            self.logger.warning(f"LLM 返回无效 JSON，使用空结果 | 片段: {cleaned[:150]}...")
            parsed = {"results": []}

        final = {k: v for k, v in record.items() if k != "results"}
        final["results"] = parsed.get("results", [])
        return final

    def _build_fallback_record(self, record: Dict) -> Dict:
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
    
    def load_model(self):
        """
        加载关系提取模型
        
        Returns:
            object: 加载的模型对象
        """
        self.logger.info(f"加载关系提取模型: {self.model_name}")
        # 模型加载逻辑将在这里实现
        # ...
        return None
    
    def preprocess(self, text, entities):
        """
        预处理输入文本和实体
        
        Args:
            text (str): 输入文本
            entities (list): 文本中的实体列表
        
        Returns:
            object: 预处理后的数据
        """
        # 文本和实体预处理逻辑将在这里实现
        # ...
        return text, entities
    
    def postprocess(self, predictions):
        """
        后处理模型预测结果
        
        Args:
            predictions (object): 模型预测结果
        
        Returns:
            list: 后处理后的关系列表
        """
        # 预测结果后处理逻辑将在这里实现
        # ...
        return []
