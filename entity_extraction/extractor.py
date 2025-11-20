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
            threshold (float): 实体识别的置信度阈值
        """
        self.model_name = model_name
        self.threshold = threshold
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"初始化实体提取器，模型: {model_name}，阈值: {threshold}")

    def extract(self, input_file, output_file, prompt_path):
        """
        执行实体提取任务

        Args:
            input_file (str or Path): 输入 JSON 文件路径（必须是包含对象列表的 JSON 数组）
            output_file (str or Path): 输出 JSONL 文件路径（每行一个结果对象）
            prompt_path (str or Path): Prompt 文件路径，用于指导 LLM 输出格式

        Returns:
            dict: 包含状态和输出路径的结果字典
        """
        input_file = Path(input_file)
        output_file = Path(output_file)
        prompt_path = Path(prompt_path)

        self.logger.info(f"开始从 {input_file} 提取实体，使用模型: {self.model_name}, Prompt: {prompt_path}")
        
        # 验证输入文件
        if not input_file.is_file():
            raise FileNotFoundError(f"输入文件不存在: {input_file}")
        
        # 验证 prompt 文件
        if not prompt_path.is_file():
            raise FileNotFoundError(f"Prompt 文件不存在: {prompt_path}")

        # 创建输出目录
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # 加载 prompt
        with open(prompt_path, 'r', encoding='utf-8') as f:
            system_prompt = f.read().strip()

        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"输入文件不是有效的 JSON 格式: {e}")

        if not isinstance(data, list):
            raise ValueError(f"输入 JSON 文件的根元素必须是一个数组（list），但得到: {type(data).__name__}")

        if not data:
            self.logger.warning(f"输入文件 {input_file} 中的数组为空")

        processed = 0
        with open(output_file, 'w', encoding='utf-8') as out_f:
            for idx, record in enumerate(data):
                if not isinstance(record, dict):
                    self.logger.warning(f"跳过非字典项（索引 {idx}）")
                    continue
            
                # 兼容 name / title，并确保有值
                name = record.get("name") or record.get("title") or f"unnamed_{idx}"
            
                content = record.get("content", "")
                if not content or (isinstance(content, str) and not content.strip()):
                    self.logger.warning(f"记录 {name} 内容为空，跳过")
                    continue
            
                # 标准化 content：过滤空值和 "---"
                if isinstance(content, list):
                    content_str = " ".join(
                        str(x) for x in content 
                        if x is not None and str(x).strip() and str(x).strip() != "---"
                    )
                else:
                    content_str = str(content).strip()
            
                if not content_str:
                    self.logger.warning(f"记录 {name} 标准化后内容为空，跳过")
                    continue
            
                full_input = {
                    "name": name,
                    "content": content_str
                }
            
                user_input = json.dumps(full_input, ensure_ascii=False, indent=2)
                full_prompt = f"{system_prompt}\n\n---\n\n{user_input}"

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
                    self.logger.error(f"处理 {name} 时出错: {e}", exc_info=True)
                    entities = []

                final_record = record.copy()
                final_record["results"] = entities
                out_f.write(json.dumps(final_record, ensure_ascii=False) + '\n')
                processed += 1

        self.logger.info(f"实体提取完成，共处理 {processed} 条记录，结果保存至 {output_file}")
        return {
            "status": "success",
            "output_path": str(output_file.resolve()),
            "processed": processed
        }

    def load_model(self):
        """
        加载实体提取模型
        
        Returns:
            object: 加载的模型对象
        """
        self.logger.info(f"加载实体提取模型: {self.model_name}")
        # 模型加载逻辑将在这里实现
        # ...
        return None
    
    def preprocess(self, text):
        """
        预处理输入文本
        
        Args:
            text (str): 输入文本
        
        Returns:
            object: 预处理后的文本
        """
        # 文本预处理逻辑将在这里实现
        # ...
        return text
    
    def postprocess(self, predictions):
        """
        后处理模型预测结果
        
        Args:
            predictions (object): 模型预测结果
        
        Returns:
            list: 后处理后的实体列表
        """
        # 预测结果后处理逻辑将在这里实现
        # ...
        return []
