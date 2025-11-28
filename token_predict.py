import re
import json
import os
import argparse
import sys
from typing import Dict, List, Union

class TaskTokenCalculator:
    
    # ==================== 实体提取 ====================
    def entity_extraction_estimate(self, content_length: int, model_name: str) -> Dict[str, float]:
        """
        实体提取任务token估算
        """
        # 固定参数
        prompt_length = 11498
        avg_entity_chars = 52
        content_per_entity = 10.4
        
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
        """
        # 固定参数
        prompt_length = 15697
        avg_triple_chars = 75
        content_per_triple = 12.9
        
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

    def entity_alignment_concept_clustering_estimate(self, content_length, model_name, x_values=None):
        """
        实体对齐和概念聚类任务token估算
        """
        if x_values is None:
            # 区间估计模式
            min_x = [0, 0, 0]
            max_x = [0.5, 0.5, 0.5]
            
            # 计算三种词的数量
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
                return None
            
            return {
                'entity_alignment': {'min_tokens': task1_min_tokens, 'max_tokens': task1_max_tokens},
                'concept_clustering': {'min_tokens': task2_min_tokens, 'max_tokens': task2_max_tokens}
            }
        
        else:
            # 精确估计模式
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
                return None
            
            return {
                'entity_alignment': {'total_tokens': task1_tokens},
                'concept_clustering': {'total_tokens': task2_tokens}
            }

    def batch_estimate(self, directory_path: str, model_name: str = "deepseek", x_values: List[float] = None) -> Dict[str, Union[Dict, List]]:
        """
        批量估算目录中JSON文件的token消耗
        
        Args:
            directory_path: 包含JSON文件的目录路径
            model_name: 模型名称 ("deepseek" 或 "qianwen")
            x_values: 用于实体消歧和概念聚类的X值列表 [x1, x2, x3]
        
        Returns:
            包含所有任务token估算结果的字典
        """
        # 检查目录是否存在
        if not os.path.exists(directory_path):
            print(f"错误: 目录 '{directory_path}' 不存在")
            return {}
        
        # 获取目录中的所有JSON文件
        json_files = [f for f in os.listdir(directory_path) if f.endswith('.json')]
        
        if not json_files:
            print(f"目录 '{directory_path}' 中没有找到JSON文件")
            return {}
        
        print(f"找到 {len(json_files)} 个JSON文件，开始处理...")
        print("=" * 60)
        
        # 计算所有文件的总字符数
        total_content_length = 0
        
        for json_file in json_files:
            file_path = os.path.join(directory_path, json_file)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 处理不同的数据结构
                content_length = 0
                
                # 如果数据是列表，遍历列表中的每个元素
                if isinstance(data, list):
                    for item in data:
                        # 如果列表元素是字典，尝试获取content
                        if isinstance(item, dict):
                            content = item.get('content', '')
                            # 移除空格，保留换行符
                            content_without_spaces = re.sub(r'[^\S\n]', '', content)
                            content_length += len(content_without_spaces)
                        # 如果列表元素是字符串，直接使用
                        elif isinstance(item, str):
                            content_without_spaces = re.sub(r'[^\S\n]', '', item)
                            content_length += len(content_without_spaces)
                # 如果数据是字典，直接获取content
                elif isinstance(data, dict):
                    content = data.get('content', '')
                    content_without_spaces = re.sub(r'[^\S\n]', '', content)
                    content_length = len(content_without_spaces)
                # 如果数据是字符串，直接使用
                elif isinstance(data, str):
                    content_without_spaces = re.sub(r'[^\S\n]', '', data)
                    content_length = len(content_without_spaces)
                else:
                    print(f"文件 {json_file} 的数据格式不支持: {type(data)}")
                    continue
                
                print(f"处理文件: {json_file}, 内容字符数: {content_length}")
                total_content_length += content_length
                    
            except Exception as e:
                print(f"处理文件 {json_file} 时出错: {e}")
        
        print(f"\n所有文件总字符数: {total_content_length}")
        print("=" * 60)
        
        # 调用三个估算函数
        entity_extraction_result = self.entity_extraction_estimate(total_content_length, model_name)
        relation_extraction_result = self.relation_extraction_estimate(total_content_length, model_name)
        alignment_clustering_result = self.entity_alignment_concept_clustering_estimate(total_content_length, model_name, x_values)
        
        # 整理结果
        results = {
            'total_content_length': total_content_length,
            'model': model_name,
            'entity_extraction': entity_extraction_result,
            'relation_extraction': relation_extraction_result
        }
        
        # 处理实体对齐和概念聚类结果
        if alignment_clustering_result:
            results['entity_alignment'] = alignment_clustering_result.get('entity_alignment', {})
            results['concept_clustering'] = alignment_clustering_result.get('concept_clustering', {})
        
        # 计算总token数
        if x_values is None:
            # 区间模式
            total_min = 0
            total_max = 0
            
            if entity_extraction_result:
                total_min += entity_extraction_result.get('total_tokens', 0)
                total_max += entity_extraction_result.get('total_tokens', 0)
            
            if relation_extraction_result:
                total_min += relation_extraction_result.get('total_tokens', 0)
                total_max += relation_extraction_result.get('total_tokens', 0)
            
            if alignment_clustering_result:
                entity_alignment = alignment_clustering_result.get('entity_alignment', {})
                concept_clustering = alignment_clustering_result.get('concept_clustering', {})
                
                total_min += entity_alignment.get('min_tokens', 0) + concept_clustering.get('min_tokens', 0)
                total_max += entity_alignment.get('max_tokens', 0) + concept_clustering.get('max_tokens', 0)
            
            results['total_tokens_range'] = {
                'min': total_min,
                'max': total_max
            }
        else:
            # 精确模式
            total_tokens = 0
            
            if entity_extraction_result:
                total_tokens += entity_extraction_result.get('total_tokens', 0)
            
            if relation_extraction_result:
                total_tokens += relation_extraction_result.get('total_tokens', 0)
            
            if alignment_clustering_result:
                entity_alignment = alignment_clustering_result.get('entity_alignment', {})
                concept_clustering = alignment_clustering_result.get('concept_clustering', {})
                
                total_tokens += entity_alignment.get('total_tokens', 0) + concept_clustering.get('total_tokens', 0)
            
            results['total_tokens'] = total_tokens
            results['x_values'] = x_values
        
        # 打印结果
        self._print_results(results, x_values is not None)
        
        return results
    
    def _print_results(self, results: Dict, has_x_values: bool):
        """打印估算结果"""
        print("\n" + "=" * 60)
        print("TOKEN估算结果汇总")
        print("=" * 60)
        
        print(f"总内容字符数: {results['total_content_length']}")
        print(f"使用模型: {results['model']}")
        
        # 打印各任务结果
        print("\n各任务估算结果:")
        print("-" * 40)
        
        # 实体提取
        if results.get('entity_extraction'):
            ee = results['entity_extraction']
            print(f"实体提取:")
            print(f"  输入token: {ee.get('input_tokens', 0):.2f}")
            print(f"  输出token: {ee.get('output_tokens', 0):.2f}")
            print(f"  总token: {ee.get('total_tokens', 0):.2f}")
        
        # 关系提取
        if results.get('relation_extraction'):
            re = results['relation_extraction']
            print(f"关系提取:")
            print(f"  输入token: {re.get('input_tokens', 0):.2f}")
            print(f"  输出token: {re.get('output_tokens', 0):.2f}")
            print(f"  总token: {re.get('total_tokens', 0):.2f}")
        
        # 实体对齐和概念聚类
        if results.get('entity_alignment') and results.get('concept_clustering'):
            ea = results['entity_alignment']
            cc = results['concept_clustering']
            
            print(f"实体对齐和概念聚类:")
            
            if has_x_values:
                print(f"  实体对齐token: {ea.get('total_tokens', 0):.2f}")
                print(f"  概念聚类token: {cc.get('total_tokens', 0):.2f}")
            else:
                print(f"  实体对齐token范围: {ea.get('min_tokens', 0):.2f} ~ {ea.get('max_tokens', 0):.2f}")
                print(f"  概念聚类token范围: {cc.get('min_tokens', 0):.2f} ~ {cc.get('max_tokens', 0):.2f}")
        
        # 打印总token数
        print("\n" + "-" * 40)
        if has_x_values:
            print(f"总token数: {results.get('total_tokens', 0):.2f}")
            print(f"使用的X值: {results.get('x_values', [])}")
        else:
            total_range = results.get('total_tokens_range', {})
            print(f"总token数范围: {total_range.get('min', 0):.2f} ~ {total_range.get('max', 0):.2f}")


def main():
    """
    批量估算token的主函数
    """
    # 直接使用用户提供的参数
    directory_path = r"E:\KKGG\data\WIKI"
    model_name = "qianwen"
    
    # 验证目录是否存在
    if not os.path.exists(directory_path):
        print(f"错误: 目录 '{directory_path}' 不存在")
        sys.exit(1)
    
    # 创建计算器并执行估算
    calculator = TaskTokenCalculator()
    
    try:
        print(f"开始估算目录: {directory_path}")
        print(f"使用模型: {model_name}")
        print("=" * 60)
        
        # 执行估算（不提供X值，返回范围）
        results = calculator.batch_estimate(
            directory_path=directory_path,
            model_name=model_name,
            x_values=None  # 不提供X值，返回范围
        )
        
        # 检查是否有结果
        if not results:
            print("没有生成任何估算结果")
            sys.exit(1)
            
    except Exception as e:
        print(f"估算过程中出现错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()