import re
from typing import Dict, List, Tuple

class TaskTokenCalculator:

    
    def count_chars(self, text: str) -> Tuple[int, int]:
        """统计中英文字符数量"""
        # 中文字符范围
        chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
        chinese_chars = len(chinese_pattern.findall(text))
        
        # 英文字符（包括数字、符号等非中文字符）
        english_chars = len(text) - chinese_chars
        
        return chinese_chars, english_chars
    
    def calculate_tokens(self, text: str) -> float:
        """计算文本的token数量"""
        chinese_chars, english_chars = self.count_chars(text)
        return chinese_chars * self.token_rates['chinese'] + english_chars * self.token_rates['english']
    
    # ==================== 实体提取 ====================
    def entity_extraction_estimate(self, input_text: str, extraction_config: Dict = None) -> Dict[str, float]:
        """
        实体提取任务token估算
        
        Args:
            input_text: 输入文本
            extraction_config: 提取配置 [待补充配置参数]
        
        Returns:
            包含输入、输出和总token的字典
        """
        # 输入token计算
        input_tokens = self.calculate_tokens(input_text)
        
        # 输出token估算规则: [待补充实体提取输出估算规则]
        output_tokens = 0  # 待实现
        
        return {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens,
            'task': 'entity_extraction'
        }
    
    # ==================== 三元组提取 ====================
    def triple_extraction_estimate(self, input_text: str, triple_config: Dict = None) -> Dict[str, float]:
        """
        三元组提取任务token估算
        
        Args:
            input_text: 输入文本
            triple_config: 三元组配置 [待补充配置参数]
        
        Returns:
            包含输入、输出和总token的字典
        """
        # 输入token计算
        input_tokens = self.calculate_tokens(input_text)
        
        # 输出token估算规则: [待补充三元组提取输出估算规则]
        output_tokens = 0  # 待实现
        
        return {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens,
            'task': 'triple_extraction'
        }
    
    # ==================== 实体削齐+概念聚类 ====================
    def entity_alignment_concept_clustering_estimate(self, content_length, model_name, x_values=[0.344, 0.183, 0.411]):
        """
        计算实体消歧和概念聚类任务的API token数
        
        Args:
            content_length: 内容字数
            model_name: 模型名称，只能是"qianwen"或"deepseek"
            x_values: 三个X值数组，默认为[0.344, 0.183, 0.411]
        
        Returns:
            两个任务的总token数
        """
        # 计算三种词的数量
        term_count_1 = content_length / 11.78  # 第一组词数量
        term_count_2 = content_length / 9.35   # 第二组词数量
        term_count_3 = content_length / 15.92  # 第三组词数量
        
        # 计算每种词每个术语的API通讯量
        term_api_1 = 102.52 + 81 * x_values[0]  # 第一组词每个术语的API通讯量
        term_api_2 = 102.52 + 81 * x_values[1]  # 第二组词每个术语的API通讯量
        term_api_3 = 102.52 + 81 * x_values[2]  # 第三组词每个术语的API通讯量
        
        # 计算每种词的总API通讯量
        total_api_1 = term_api_1 * term_count_1  # 第一组词总API通讯量
        total_api_2 = term_api_2 * term_count_2  # 第二组词总API通讯量
        total_api_3 = term_api_3 * term_count_3  # 第三组词总API通讯量
        
        # 计算任务总字数
        task1_total_chars = total_api_1 + total_api_3 + 2 * content_length  # 实体消歧任务总字数
        task2_total_chars = total_api_2 + content_length  # 概念聚类任务总字数
        
        # 根据模型计算token数
        if model_name.lower() == "deepseek":
            task1_tokens = task1_total_chars * 0.6
            task2_tokens = task2_total_chars * 0.6
        elif model_name.lower() == "qianwen":
            task1_tokens = task1_total_chars * 1.0
            task2_tokens = task2_total_chars * 1.0
        else:
            print(f"实体消歧任务总字数: {task1_total_chars:.2f}")
            print(f"概念聚类任务总字数: {task2_total_chars:.2f}")
            print("模型名称无效，请输入'qianwen'或'deepseek'")
            return None, None
        
        # 打印结果
        print(f"实体消歧任务总token数: {task1_tokens:.2f}")
        print(f"概念聚类任务总token数: {task2_tokens:.2f}")
        
        return task1_tokens, task2_tokens
    

    # ==================== 质量评估 ====================
    def quality_assessment_estimate(self, input_text: str, assessment_config: Dict = None) -> Dict[str, float]:
        """
        质量评估任务token估算
        
        Args:
            input_text: 输入文本或需要评估的内容
            assessment_config: 评估配置 [待补充配置参数]
        
        Returns:
            包含输入、输出和总token的字典
        """
        # 输入token计算
        input_tokens = self.calculate_tokens(input_text)
        
        # 输出token估算规则: [待补充质量评估输出估算规则]
        output_tokens = 0  # 待实现
        
        return {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens,
            'task': 'quality_assessment'
        }
    
    # ==================== 批量估算 ====================
    def batch_estimate(self, tasks: List[Dict]) -> List[Dict]:
        """
        批量估算多个任务的token
        
        Args:
            tasks: 任务列表，每个任务包含task_type和相应参数
        
        Returns:
            各任务的token估算结果列表
        """
        results = []
        
        for task in tasks:
            task_type = task.get('task_type')
            params = task.get('params', {})
            
            if task_type == 'entity_extraction':
                result = self.entity_extraction_estimate(
                    params.get('input_text', ''), 
                    params.get('config')
                )
            elif task_type == 'triple_extraction':
                result = self.triple_extraction_estimate(
                    params.get('input_text', ''), 
                    params.get('config')
                )
            elif task_type == 'entity_alignment':
                result = self.entity_alignment_estimate(
                    params.get('entities', []), 
                    params.get('config')
                )
            elif task_type == 'concept_clustering':
                result = self.concept_clustering_estimate(
                    params.get('concepts', []), 
                    params.get('config')
                )
            elif task_type == 'quality_assessment':
                result = self.quality_assessment_estimate(
                    params.get('input_text', ''), 
                    params.get('config')
                )
            else:
                continue
                
            results.append(result)
        
        return results


# 使用示例
if __name__ == "__main__":
    calculator = TaskTokenCalculator()
    
    # 测试文本
    test_text = "这是一段测试文本，包含中文和English words。"
    
    # 计算基础token
    tokens = calculator.calculate_tokens(test_text)
    print(f"测试文本token数量: {tokens}")
    
    # 各任务估算示例（输出部分需要您补充规则）
    print("\n=== 各任务Token估算 ===")
    
    # 实体提取
    entity_result = calculator.entity_extraction_estimate(test_text)
    print(f"实体提取: {entity_result}")
    
    # 三元组提取
    triple_result = calculator.triple_extraction_estimate(test_text)
    print(f"三元组提取: {triple_result}")
    
    # 实体削齐
    entities = ["实体1", "entity2", "测试实体"]
    alignment_result = calculator.entity_alignment_estimate(entities)
    print(f"实体削齐: {alignment_result}")
    
    # 概念聚类
    concepts = ["概念A", "concept B", "测试概念"]
    clustering_result = calculator.concept_clustering_estimate(concepts)
    print(f"概念聚类: {clustering_result}")
    
    # 质量评估
    assessment_result = calculator.quality_assessment_estimate(test_text)
    print(f"质量评估: {assessment_result}")
    
    # 批量估算示例
    print("\n=== 批量估算 ===")
    batch_tasks = [
        {
            'task_type': 'entity_extraction',
            'params': {'input_text': test_text}
        },
        {
            'task_type': 'triple_extraction', 
            'params': {'input_text': test_text}
        }
    ]
    
    batch_results = calculator.batch_estimate(batch_tasks)
    for result in batch_results:
        print(f"{result['task']}: {result['total_tokens']} tokens")