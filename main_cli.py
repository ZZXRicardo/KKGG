#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
主要命令行工具，用于执行图谱构建的各项任务
包括：实体提取、关系提取、实体消歧、局部概念聚类
"""

import argparse
import logging
import os
import json
from pathlib import Path
from typing import List, Dict, Set, Tuple
from cli import CLI
#from EntityExtractor import EntityExtractor
#from relation_extraction.extractor import RelationExtractor
from disambiguator_clusterer import TermDisambiguator



class MainCLI(CLI):
    """图谱构建主命令行工具，提供实体提取、关系提取、实体消歧、局部概念聚类等功能"""
    
    def _add_arguments(self):
        """添加命令行参数"""
        super()._add_arguments()
        
        # 添加任务选择参数
        self.parser.add_argument('--task', type=str, required=True,
                                choices=['entity_extraction', 'relation_extraction', 
                                         'entity_disambiguation', 'concept_clustering', 'all'],
                                help='要执行的任务')
        
        # 实体提取相关参数
        self.parser.add_argument('--entity_model', type=str, default='default',
                                help='实体提取使用的模型')
        self.parser.add_argument('--entity_threshold', type=float, default=0.5,
                                help='实体提取的置信度阈值')
        
        # 关系提取相关参数
        self.parser.add_argument('--relation_model', type=str, default='default',
                                help='关系提取使用的模型')
        self.parser.add_argument('--relation_threshold', type=float, default=0.5,
                                help='关系提取的置信度阈值')
        
        # 实体消歧相关参数
        self.parser.add_argument('--disambiguation_method', type=str, default='default',
                                help='实体消歧使用的方法')
        self.parser.add_argument('--kb_path', type=str, default='',
                                help='知识库路径，用于实体消歧')
        
        # 概念聚类相关参数
        self.parser.add_argument('--clustering_method', type=str, default='default',
                                help='概念聚类使用的方法')
        self.parser.add_argument('--cluster_num', type=int, default=-1,
                                help='聚类数量，-1表示自动确定')
    
    def run(self, args):
        """
        执行选定的任务
        
        Args:
            args: 解析后的命令行参数
        """
        logging.info(f"开始执行任务: {args.task}")
        
        if args.task == 'all':
            self._run_entity_extraction(args)
            self._run_relation_extraction(args)
            self._run_entity_disambiguation(args)
            self._run_concept_clustering(args)
        elif args.task == 'entity_extraction':
            self._run_entity_extraction(args)
        elif args.task == 'relation_extraction':
            self._run_relation_extraction(args)
        elif args.task == 'entity_disambiguation':
            self._run_entity_disambiguation(args)
        elif args.task == 'concept_clustering':
            self._run_concept_clustering(args)
        
        logging.info(f"任务 {args.task} 执行完成")
    
    def _run_entity_extraction(self, args):
        """
        执行实体提取任务
        
        Args:
            args: 命令行参数
        """
        logging.info("开始实体提取任务")
        extractor = EntityExtractor(
            model_name=args.entity_model,
            threshold=args.entity_threshold
        )
        extractor.extract(
            data_dir=args.data_dir,
            output_dir=args.output_dir
        )
        logging.info("实体提取任务完成")
    
    def _run_relation_extraction(self, args):
        """
        执行关系提取任务
        
        Args:
            args: 命令行参数
        """
        logging.info("开始关系提取任务")
        extractor = RelationExtractor(
            model_name=args.relation_model,
            threshold=args.relation_threshold
        )
        extractor.extract(
            data_dir=args.data_dir,
            output_dir=args.output_dir
        )
        logging.info("关系提取任务完成")
    
    def _run_entity_disambiguation(self, args):
        """
        执行实体消歧任务
        
        从输出.json中：
        1. 遍历relations数组，识别所有词的label属性
        2. 收集所有label=='a'的词（不论在head还是tail位置）
        3. 收集所有关系词（triple[1]）
        4. 调用Disambiguate函数进行消歧
        5. 将更新后的词替换回原三元组
        
        Args:
            args: 命令行参数
        """
        logging.info("开始实体消歧任务")
        
        # 定义输入输出路径
        input_json_path = r"E:\KKGG\output\KG\输出.json"
        
        # 读取JSON数据
        try:
            with open(input_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logging.info(f"成功读取输入文件: {input_json_path}")
        except Exception as e:
            logging.error(f"读取输入文件失败: {e}")
            return
        
        # 收集所有label=='a'的实体（不论位置）和所有关系词
        entity_a_terms = set()      # label=='a'的实体（head或tail）
        relation_terms = set()       # 所有关系词
        entity_context_parts = []    # 用于构建实体上下文
        relation_context_parts = []  # 用于构建关系上下文
        
        for item in data:
            if 'results' not in item:
                continue
                
            for result in item['results']:
                if 'output' not in result:
                    continue
                
                output = result['output']
                
                # 从relations数组中提取
                if 'relations' in output:
                    for rel in output['relations']:
                        if not isinstance(rel, dict):
                            continue
                        
                        triple = rel.get('triple', [])
                        labels = rel.get('label', [])
                        
                        if len(triple) >= 3 and len(labels) >= 2:
                            head = triple[0]
                            relation = triple[1]
                            tail = triple[2]
                            head_label = labels[0]  # head的label
                            tail_label = labels[1]  # tail的label
                            
                            # 1. 检查head的label，如果是'a'则收集
                            if head_label and str(head_label).lower() == 'a' and head:
                                entity_a_terms.add(head)
                                entity_context_parts.append(f"{head} {relation} {tail}")
                                logging.debug(f"找到label='a'的实体(head): {head}")
                            
                            # 2. 检查tail的label，如果是'a'则收集
                            if tail_label and str(tail_label).lower() == 'a' and tail:
                                entity_a_terms.add(tail)
                                entity_context_parts.append(f"{head} {relation} {tail}")
                                logging.debug(f"找到label='a'的实体(tail): {tail}")
                            
                            # 3. 收集所有关系词
                            if relation:
                                relation_terms.add(relation)
                            
                            # 4. 构建关系上下文
                            relation_context_parts.append(f"{head} {relation} {tail}")
        
        # 转换为列表
        entity_a_list = sorted(list(entity_a_terms))
        relation_list = sorted(list(relation_terms))
        
        logging.info(f"收集到 {len(entity_a_list)} 个label='a'的实体（不论位置）")
        logging.info(f"收集到 {len(relation_list)} 个关系词")
        logging.info(f"实体样例: {entity_a_list[:5]}")
        logging.info(f"关系样例: {relation_list[:5]}")
        
        if not entity_a_list and not relation_list:
            logging.warning("未找到任何实体或关系词，跳过消歧")
            return
        
        # 构建共享上下文（限制长度避免过长）
        entity_shared_context = " ".join(entity_context_parts[:100])
        relation_shared_context = " ".join(relation_context_parts[:100])
        
        # 初始化消歧器并执行消歧
        disambiguator = TermDisambiguator(api_provider="qianwen")
        
        try:
            # ✅ 修复：只传入方法定义中接受的参数
            updated_entities, updated_relations = disambiguator.Disambiguate(
                entity_terms=entity_a_list,
                relation_terms=relation_list,
                entity_shared_context=entity_shared_context,
                relation_shared_context=relation_shared_context
            )
            
            logging.info(f"消歧完成 - 更新后实体数: {len(updated_entities)}, 关系数: {len(updated_relations)}")
            
        except Exception as e:
            logging.error(f"消歧过程出错: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return
        
        # 创建映射字典：原术语 -> 更新后术语
        entity_mapping = dict(zip(entity_a_list, updated_entities))
        relation_mapping = dict(zip(relation_list, updated_relations))
        
        logging.info(f"实体映射示例: {dict(list(entity_mapping.items())[:3])}")
        logging.info(f"关系映射示例: {dict(list(relation_mapping.items())[:3])}")
        
        # 更新原JSON数据中的三元组
        updated_count = 0
        for item in data:
            if 'results' not in item:
                continue
                
            for result in item['results']:
                if 'output' not in result or 'relations' not in result['output']:
                    continue
                
                relations = result['output']['relations']
                for rel in relations:
                    if not isinstance(rel, dict) or 'triple' not in rel:
                        continue
                    
                    triple = rel['triple']
                    if len(triple) >= 3:
                        head, relation, tail = triple[0], triple[1], triple[2]
                        
                        # 更新head（如果在entity_mapping中）
                        if head in entity_mapping and entity_mapping[head] != head:
                            rel['triple'][0] = entity_mapping[head]
                            updated_count += 1
                            logging.debug(f"更新head: {head} -> {entity_mapping[head]}")
                        
                        # 更新relation（如果在relation_mapping中）
                        if relation in relation_mapping and relation_mapping[relation] != relation:
                            rel['triple'][1] = relation_mapping[relation]
                            updated_count += 1
                            logging.debug(f"更新relation: {relation} -> {relation_mapping[relation]}")
                        
                        # 更新tail（如果在entity_mapping中）
                        if tail in entity_mapping and entity_mapping[tail] != tail:
                            rel['triple'][2] = entity_mapping[tail]
                            updated_count += 1
                            logging.debug(f"更新tail: {tail} -> {entity_mapping[tail]}")
        
        # 保存更新后的JSON
        output_json_path = os.path.join(args.output_dir, "输出_消歧后.json")
        try:
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logging.info(f"更新后的数据已保存至: {output_json_path}")
            logging.info(f"共更新了 {updated_count} 个术语")
        except Exception as e:
            logging.error(f"保存更新后的文件失败: {e}")
        
        logging.info("实体消歧任务完成")
    
    def _run_concept_clustering(self, args):
        """
        执行局部概念聚类任务
        
        从输出.json中：
        1. 遍历relations数组，识别所有词的label属性
        2. 收集所有label=='b'的词（不论在head还是tail位置）
        3. 调用clusterer函数进行聚类（不替换原文件）
        
        Args:
            args: 命令行参数
        """
        logging.info("开始局部概念聚类任务")
        
        # 定义输入路径
        input_json_path = r"E:\KKGG\output\KG\输出.json"
        
        # 读取JSON数据
        try:
            with open(input_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logging.info(f"成功读取输入文件: {input_json_path}")
        except Exception as e:
            logging.error(f"读取输入文件失败: {e}")
            return
        
        # 收集所有label=='b'的实体（不论位置）
        entity_b_terms = set()   # label=='b'的实体（head或tail）
        context_parts = []        # 用于构建共享上下文
        
        for item in data:
            if 'results' not in item:
                continue
                
            for result in item['results']:
                if 'output' not in result:
                    continue
                
                output = result['output']
                
                # 从relations数组中提取
                if 'relations' in output:
                    for rel in output['relations']:
                        if not isinstance(rel, dict):
                            continue
                        
                        triple = rel.get('triple', [])
                        labels = rel.get('label', [])
                        
                        if len(triple) >= 3 and len(labels) >= 2:
                            head = triple[0]
                            relation = triple[1]
                            tail = triple[2]
                            head_label = labels[0]  # head的label
                            tail_label = labels[1]  # tail的label
                            
                            # 1. 检查head的label，如果是'b'则收集
                            if head_label and str(head_label).lower() == 'b' and head:
                                entity_b_terms.add(head)
                                context_parts.append(f"{head} {relation} {tail}")
                                logging.debug(f"找到label='b'的实体(head): {head}")
                            
                            # 2. 检查tail的label，如果是'b'则收集
                            if tail_label and str(tail_label).lower() == 'b' and tail:
                                entity_b_terms.add(tail)
                                context_parts.append(f"{head} {relation} {tail}")
                                logging.debug(f"找到label='b'的实体(tail): {tail}")
        
        # 转换为列表
        entity_b_list = sorted(list(entity_b_terms))
        
        logging.info(f"收集到 {len(entity_b_list)} 个label='b'的实体（不论位置）")
        logging.info(f"实体样例: {entity_b_list[:10]}")
        
        if not entity_b_list:
            logging.warning("未找到任何label='b'的实体，跳过聚类")
            return
        
        # 构建共享上下文（限制长度）
        shared_context = " ".join(context_parts[:200])
        
        # 初始化聚类器并执行聚类
        clusterer = TermDisambiguator(api_provider="qianwen")
        
        try:
            # ✅ 修复：只传入方法定义中接受的参数
            cluster_result = clusterer.clusterer(
                terms=entity_b_list,
                shared_context=shared_context
            )
            
            logging.info(f"聚类完成 - 生成的聚类结果已保存")
            logging.info(f"聚类结果包含 {len(cluster_result)} 个术语的三元组信息")
            
            # 打印部分聚类结果
            for i, (term, triples) in enumerate(list(cluster_result.items())[:3]):
                logging.info(f"术语 '{term}' 的三元组数量: {len(triples)}")
                
        except Exception as e:
            logging.error(f"聚类过程出错: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return
        
        logging.info("局部概念聚类任务完成")


if __name__ == "__main__":
    cli = MainCLI()
    cli.execute()