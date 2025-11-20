import re
from typing import Dict, List, Tuple

class TaskTokenCalculator:
    
    def __init__(self):
        self.token_rates = {
            'chinese': 2.0,  # 中文字符token比例
            'english': 0.8   # 英文字符token比例
        }
    
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
    def entity_extraction_estimate(self, content_length: int, model_name: str) -> Dict[str, float]:
        """
        实体提取任务token估算
        
        Args:
            content_length: 内容文本的字符数
            model_name: 模型名称 ("deepseek" 或 "qianwen")
        
        Returns:
            包含输入、输出和总token的字典
        """
        # 固定参数
        prompt_length = 11498  # 系统提示词字符数
        avg_entity_chars = 52  # 每个实体平均52字符
        content_per_entity = 10.4  # 每10.4字符提取1个实体
        
        # 计算实体数量
        entity_count = content_length / content_per_entity
        
        # 输入字符数 = 提示词 + content
        input_chars = prompt_length + content_length
        
        # 输出字符数 = content重复 + 实体部分
        output_chars = content_length + (avg_entity_chars * entity_count)
        
        # 根据模型计算token数
        if model_name.lower() == "deepseek":
            token_ratio = 0.6
        elif model_name.lower() == "qianwen":
            token_ratio = 0.4
        else:
            return {}
        
        input_tokens = input_chars * token_ratio
        output_tokens = output_chars * token_ratio
        total_tokens = input_tokens + output_tokens
        
        return {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': total_tokens,
            'task': 'entity_extraction'
        }
    
    # ==================== 关系提取（三元组提取） ====================
    def relation_extraction_estimate(self, content_length: int, model_name: str) -> Dict[str, float]:
        """
        关系提取任务token估算
        
        Args:
            content_length: 内容文本的字符数
            model_name: 模型名称 ("deepseek" 或 "qianwen")
        
        Returns:
            包含输入、输出和总token的字典
        """
        # 固定参数
        prompt_length = 15697  # 系统提示词字符数
        avg_triple_chars = 75  # 每个三元组平均75字符
        content_per_triple = 12.9  # 每12.9字符产生1个三元组
        
        # 计算三元组数量
        triple_count = content_length / content_per_triple
        
        # 输入字符数 = 提示词 + content
        input_chars = prompt_length + content_length
        
        # 输出字符数 = content重复 + 三元组部分
        output_chars = content_length + (avg_triple_chars * triple_count)
        
        # 根据模型计算token数
        if model_name.lower() == "deepseek":
            token_ratio = 0.6
        elif model_name.lower() == "qianwen":
            token_ratio = 0.4
        else:
            return {}
        
        input_tokens = input_chars * token_ratio
        output_tokens = output_chars * token_ratio
        total_tokens = input_tokens + output_tokens
        
        return {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': total_tokens,
            'task': 'relation_extraction'
        }
    
    # ==================== 三元组提取（兼容旧名称） ====================
    def triple_extraction_estimate(self, content_length: int, model_name: str) -> Dict[str, float]:
        """
        三元组提取任务token估算（兼容旧名称）
        """
        return self.relation_extraction_estimate(content_length, model_name)
    
    def entity_alignment_concept_clustering_estimate(self, content_length, model_name, x_values=None):
        # 如果没有提供x_values，计算区间范围
        if x_values is None:
            print(f"内容字数: {content_length}")
            print("计算X值在[0, 1]范围内的区间估计:")
            print("-" * 50)
            
            # 计算最小值和最大值（x=0和x=1）
            min_x = [0, 0, 0]
            max_x = [0.5, 0.5, 0.5]
            
            # 计算三种词的数量（保持不变）
            term_count_1 = content_length / 13.2
            term_count_2 = content_length / 12.9
            term_count_3 = content_length / 12.5
            
            # 计算最小和最大API通讯量
            term_api_min_1 = 102.52 + 81 * min_x[0]
            term_api_min_2 = 102.52 + 81 * min_x[1] 
            term_api_min_3 = 102.52 + 81 * min_x[2]
            
            term_api_max_1 = 102.52 + 81 * max_x[0]
            term_api_max_2 = 102.52 + 81 * max_x[1]
            term_api_max_3 = 102.52 + 81 * max_x[2]
            
            # 计算最小和最大总API通讯量
            total_api_min_1 = term_api_min_1 * term_count_1
            total_api_min_2 = term_api_min_2 * term_count_2
            total_api_min_3 = term_api_min_3 * term_count_3
            
            total_api_max_1 = term_api_max_1 * term_count_1
            total_api_max_2 = term_api_max_2 * term_count_2
            total_api_max_3 = term_api_max_3 * term_count_3
            
            # 计算任务总字数范围
            task1_min_chars = total_api_min_1 + total_api_min_3 + 2 * content_length
            task1_max_chars = total_api_max_1 + total_api_max_3 + 2 * content_length
            
            task2_min_chars = total_api_min_2 + content_length
            task2_max_chars = total_api_max_2 + content_length
            
            # 根据模型计算token数范围
            if model_name.lower() == "deepseek":
                task1_min_tokens = task1_min_chars * 0.6
                task1_max_tokens = task1_max_chars * 0.6
                task2_min_tokens = task2_min_chars * 0.6
                task2_max_tokens = task2_max_chars * 0.6
            elif model_name.lower() == "qianwen":
                task1_min_tokens = task1_min_chars * 0.4
                task1_max_tokens = task1_max_chars * 0.4
                task2_min_tokens = task2_min_chars * 0.4
                task2_max_tokens = task2_max_chars * 0.4
            else:
                print("模型名称无效，请输入'qianwen'或'deepseek'")
                return None, None
            
            # 打印区间结果
            print(f"实体消歧任务token数范围: {task1_min_tokens:.2f} ~ {task1_max_tokens:.2f}")
            print(f"概念聚类任务token数范围: {task2_min_tokens:.2f} ~ {task2_max_tokens:.2f}")
            print(f"总token数范围: {task1_min_tokens + task2_min_tokens:.2f} ~ {task1_max_tokens + task2_max_tokens:.2f}")
            
            return (task1_min_tokens, task1_max_tokens), (task2_min_tokens, task2_max_tokens)
        
        else:
            # 原有逻辑保持不变
            # 计算三种词的数量
            term_count_1 = content_length / 11.78
            term_count_2 = content_length / 9.35
            term_count_3 = content_length / 15.92
            
            # 计算每种词每个术语的API通讯量
            term_api_1 = 102.52 + 81 * x_values[0]
            term_api_2 = 102.52 + 81 * x_values[1]
            term_api_3 = 102.52 + 81 * x_values[2]
            
            # 计算每种词的总API通讯量
            total_api_1 = term_api_1 * term_count_1
            total_api_2 = term_api_2 * term_count_2
            total_api_3 = term_api_3 * term_count_3
            
            # 计算任务总字数
            task1_total_chars = total_api_1 + total_api_3 + 2 * content_length
            task2_total_chars = total_api_2 + content_length
            
            # 根据模型计算token数
            if model_name.lower() == "deepseek":
                task1_tokens = task1_total_chars * 0.6
                task2_tokens = task2_total_chars * 0.6
            elif model_name.lower() == "qianwen":
                task1_tokens = task1_total_chars * 0.4
                task2_tokens = task2_total_chars * 0.4
            else:
                print(f"实体消歧任务总字数: {task1_total_chars:.2f}")
                print(f"概念聚类任务总字数: {task2_total_chars:.2f}")
                print("模型名称无效，请输入'qianwen'或'deepseek'")
                return None, None
            
            # 打印结果
            print(f"实体消歧任务总token数: {task1_tokens:.2f}")
            print(f"概念聚类任务总token数: {task2_tokens:.2f}")
            
            return task1_tokens, task2_tokens
 
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
                    params.get('content_length', 0), 
                    params.get('model_name', 'deepseek')
                )
            elif task_type == 'relation_extraction' or task_type == 'triple_extraction':
                result = self.relation_extraction_estimate(
                    params.get('content_length', 0), 
                    params.get('model_name', 'deepseek')
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
    
    # 各任务估算示例
    print("\n=== 各任务Token估算 ===")
    
    # 实体提取
    entity_result = calculator.entity_extraction_estimate(2000, "deepseek")
    print(f"实体提取: {entity_result}")
    
    # 关系提取
    relation_result = calculator.relation_extraction_estimate(2000, "deepseek")
    print(f"关系提取: {relation_result}")
    
    # 三元组提取（兼容旧名称）
    triple_result = calculator.triple_extraction_estimate(2000, "deepseek")
    print(f"三元组提取: {triple_result}")
    
    # 批量估算示例
    print("\n=== 批量估算 ===")
    batch_tasks = [
        {
            'task_type': 'entity_extraction',
            'params': {'content_length': 2000, 'model_name': 'deepseek'}
        },
        {
            'task_type': 'relation_extraction', 
            'params': {'content_length': 2000, 'model_name': 'deepseek'}
        }
    ]
    
    batch_results = calculator.batch_estimate(batch_tasks)
    for result in batch_results:
        print(f"{result['task']}: {result['total_tokens']:.2f} tokens")