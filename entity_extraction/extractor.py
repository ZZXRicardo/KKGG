#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
实体提取器模块
负责从文本中提取实体
"""

import logging
import os
import json
from pathlib import Path
from typing import Dict, Any
from project.llm import LLM  


class EntityExtractor:
    """
    实体提取器类
    用于从文本数据中识别和提取实体
    """
    
    def __init__(self, model_name='default', threshold=0.5, system_prompt=""):
        """
        初始化实体提取器
        
        Args:
            model_name (str): 使用的模型名称（如 'qianwen', 'deepseek'）
            threshold (float): 实体识别的置信度阈值（LLM 场景下可忽略）
            system_prompt (str): 系统提示词，定义抽取规则和输出格式
        """
        self.model_name = model_name
        self.threshold = threshold
        self.system_prompt = system_prompt
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"初始化实体提取器，模型: {model_name}，阈值: {threshold}")
    
    def extract(self, data_dir: str, output_dir: str) -> Dict[str, Any]:
        """
        执行实体提取任务
        
        Args:
            data_dir (str): 输入数据目录（包含 .json 文件，每文件一条记录）
            output_dir (str): 输出结果目录
        
        Returns:
            dict: 包含状态和输出路径的结果字典
        """
        self.logger.info(f"开始从 {data_dir} 提取实体，使用模型: {self.model_name}")
        os.makedirs(output_dir, exist_ok=True)

        input_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
        if not input_files:
            self.logger.warning(f"目录 {data_dir} 中没有找到 .json 文件")
            output_path = os.path.join(output_dir, "entities_extracted.jsonl")
            return {"status": "empty", "output_path": output_path, "processed": 0}

        output_path = os.path.join(output_dir, "entities_extracted.jsonl")

        processed = 0
        with open(output_path, 'w', encoding='utf-8') as out_f:
            for filename in sorted(input_files):
                file_path = os.path.join(data_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        record = json.load(f)
                except Exception as e:
                    self.logger.error(f"跳过无效文件 {filename}: {e}")
                    continue

                content = record.get("content", "")
                name = record.get("name", Path(filename).stem)

                if not content.strip():
                    self.logger.warning(f"记录 {name} 内容为空，跳过")
                    continue
              
                full_input = {
                    "name": record.get("name"),
                    "address": record.get("address"),
                    "category": record.get("category"),
                    "content": record.get("content", "")
                }
                
                content_val = full_input["content"]
                if isinstance(content_val, list):
                    full_input["content"] = " ".join(str(x) for x in content_val if x)
                else:
                    full_input["content"] = str(content_val)
                
                user_input = json.dumps(
                    full_input,
                    ensure_ascii=False,
                    indent=2
                )
                
                full_prompt = f"{self.system_prompt}\n\n---\n\n{user_input}"

                try:
                    llm = LLM(prompt=full_prompt, api_provider=self.model_name)
                    raw_response = llm.llm_call()
                    extracted_text = llm.extract_response(raw_response)

                    try:
                        result_obj = json.loads(extracted_text)
                        entities = result_obj.get("results", [])
                    except json.JSONDecodeError:
                        self.logger.warning(f"LLM 返回非 JSON 格式，记录: {name}")
                        entities = []

                except Exception as e:
                    self.logger.error(f"处理 {name} 时出错: {e}")
                    entities = []

                final_record = record.copy()
                final_record["results"] = entities
                out_f.write(json.dumps(final_record, ensure_ascii=False) + '\n')
                processed += 1

        self.logger.info(f"实体提取完成，共处理 {processed} 条记录，结果保存至 {output_path}")
        return {
            "status": "success",
            "output_path": output_path,
            "processed": processed
        }
    
    def load_model(self):
        """
        加载实体提取模型（LLM 场景下可留空或用于验证）
        """
        self.logger.info(f"加载实体提取模型: {self.model_name}")
        return None
    
    def preprocess(self, text):
        """预处理输入文本"""
        return text.strip()
    
    def postprocess(self, predictions):
        """后处理模型预测结果"""
        return predictions
