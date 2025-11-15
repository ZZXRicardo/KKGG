#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
主要命令行工具，用于执行图谱构建的各项任务
包括：实体提取、关系提取、实体消歧、局部概念聚类
支持按文章逐个处理、增量输出和断点续处理
"""

import argparse
import logging
import os
import json
from pathlib import Path
from typing import List, Dict, Set, Tuple
from cli import CLI
from entity_extraction.extractor import EntityExtractor
from relation_extraction.extractor import RelationExtractor
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
        self.parser.add_argument('--entity_input_dir', type=str, required=True,
                                help='实体提取的输入目录）')
        self.parser.add_argument('--entity_output_dir', type=str, required=True,
                                help='实体提取的输出目录')
        self.parser.add_argument('--entity_prompt', type=str, default='default',
                                help='实体提取的 Prompt 模板路径')
        self.parser.add_argument('--entity_threshold', type=float, default=0.5,
                                help='实体提取的置信度阈值')
        
        #  关系提取相关参数 
        self.parser.add_argument('--relation_model', type=str, default='default',
                                help='关系提取使用的模型')
        self.parser.add_argument('--relation_input_dir', type=str, required=True,
                                help='关系提取的输入目录')
        self.parser.add_argument('--relation_output_dir', type=str, required=True,
                                help='关系提取的输出目录')
        self.parser.add_argument('--relation_prompt', type=str, default='default',
                                help='关系提取的 Prompt 模板路径')
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
        
        # 新增：断点续处理参数
        self.parser.add_argument('--start_index', type=int, default=0,
                                help='开始处理的文章索引（包含）')
        self.parser.add_argument('--end_index', type=int, default=-1,
                                help='结束处理的文章索引（包含），-1表示处理到末尾')
        self.parser.add_argument('--resume', action='store_true',
                                help='从进度文件恢复处理')
        self.parser.add_argument('--progress_file', type=str, 
                                default=r'E:\KKGG\project\process.json',
                                help='进度文件路径')
    
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
        执行实体提取任务（目录模式）
        """
        logging.info("开始实体提取任务")
        
        input_dir = Path(args.entity_input_dir)
        output_dir = Path(args.entity_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        extractor = EntityExtractor(
            model_name=args.entity_model,
            threshold=args.entity_threshold     
        )
        
        input_files = sorted(input_dir.glob("*.json"))
        total_articles = 0
        
        for idx, input_file in enumerate(input_files):
            if idx < args.start_index:
                continue
            if args.end_index >= 0 and idx > args.end_index:
                break
            
            output_file = output_dir / (input_file.stem + ".jsonl")
            
            if args.resume and output_file.exists():
                logging.info(f"跳过（已存在）: {output_file.name}")
                continue
            
            logging.info(f"处理: {input_file.name} -> {output_file.name}")
            count = extractor.extract_batch_to_jsonl(
                input_file=input_file,
                output_file=output_file,
                prompt_path=Path(args.entity_prompt)
            )
            total_articles += count
        
        logging.info(f"实体提取完成，共处理 {total_articles} 篇文章")
    
    def _run_relation_extraction(self, args):
        """
        执行关系提取任务（目录模式）
        """
        logging.info("开始关系提取任务")
        
        input_dir = Path(args.relation_input_dir)
        output_dir = Path(args.relation_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        extractor = RelationExtractor(
            model_name=args.relation_model,
            threshold=args.relation_threshold
        )
        
        input_files = sorted(input_dir.glob("*.jsonl"))
        total_articles = 0
        
        for idx, input_file in enumerate(input_files):
            if idx < args.start_index:
                continue
            if args.end_index >= 0 and idx > args.end_index:
                break
            
            output_file = output_dir / (input_file.stem + ".jsonl")
            
            if args.resume and output_file.exists():
                logging.info(f"跳过（已存在）: {output_file.name}")
                continue
            
            logging.info(f"处理: {input_file.name} -> {output_file.name}")
            count = extractor.extract_relations_from_jsonl(
                input_file=input_file,
                output_file=output_file,
                prompt_path=Path(args.relation_prompt)
            )
            total_articles += count
        
        logging.info(f"关系提取完成，共处理 {total_articles} 篇文章")
    
    def _ensure_progress_file_exists(self, progress_file: str):
        """确保进度文件存在且格式正确"""
        progress_dir = os.path.dirname(progress_file)
        os.makedirs(progress_dir, exist_ok=True)
        
        if not os.path.exists(progress_file):
            # 创建空的进度文件
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False, indent=2)
            logging.info(f"创建进度文件: {progress_file}")
    
    def _load_progress(self, progress_file: str, task_name: str) -> Dict:
        """加载指定任务的进度"""
        self._ensure_progress_file_exists(progress_file)
        
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                all_progress = json.load(f)
            
            # 查找指定任务的进度
            for task_progress in all_progress:
                if task_progress.get("task") == task_name:
                    return task_progress
            
            # 如果找不到指定任务的进度，返回默认值
            logging.info(f"未找到任务 {task_name} 的进度记录，使用默认值")
        except Exception as e:
            logging.warning(f"读取进度文件失败: {e}")
        
        # 返回默认进度
        return {
            "task": task_name,
            "processed_indices": [],
            "current_index": 0
        }
    
    def _save_progress(self, progress_file: str, task_progress: Dict):
        """保存指定任务的进度"""
        try:
            # 确保目录存在
            self._ensure_progress_file_exists(progress_file)
            
            # 读取所有进度
            all_progress = []
            if os.path.exists(progress_file):
                with open(progress_file, 'r', encoding='utf-8') as f:
                    all_progress = json.load(f)
            
            # 更新或添加当前任务的进度
            task_name = task_progress["task"]
            found = False
            for i, progress in enumerate(all_progress):
                if progress.get("task") == task_name:
                    all_progress[i] = task_progress
                    found = True
                    break
            
            if not found:
                all_progress.append(task_progress)
            
            # 保存所有进度
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(all_progress, f, ensure_ascii=False, indent=2)
            
            logging.debug(f"进度已保存: {progress_file} (任务: {task_name})")
        except Exception as e:
            logging.error(f"保存进度文件失败: {e}")
    
    def _run_entity_disambiguation(self, args):
        """
        执行实体消歧任务 - 按文章逐个处理
        
        Args:
            args: 命令行参数
        """
        logging.info("开始实体消歧任务（按文章逐个处理）")
        
        # 定义输入输出路径
        input_json_path = r"E:\KKGG\output\KG\输出.json"
        output_json_path = r"E:\KKGG\output\KG\输出_消歧后.json"
        progress_file = args.progress_file
        task_name = "entity_disambiguation"
        
        # ✅ 修复：检查输入文件是否存在
        if not os.path.exists(input_json_path):
            logging.error(f"输入文件不存在: {input_json_path}")
            # 创建初始进度并保存
            progress = {
                "task": task_name,
                "processed_indices": [],
                "current_index": 0
            }
            self._save_progress(progress_file, progress)
            return
        
        # 读取JSON数据
        try:
            with open(input_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logging.info(f"成功读取输入文件: {input_json_path}, 共 {len(data)} 篇文章")
        except Exception as e:
            logging.error(f"读取输入文件失败: {e}")
            # ✅ 修复：即使读取失败也保存进度
            progress = {
                "task": task_name,
                "processed_indices": [],
                "current_index": 0
            }
            self._save_progress(progress_file, progress)
            return
        
        # 加载进度
        progress = self._load_progress(progress_file, task_name)
        if args.resume:
            start_index = progress.get("current_index", args.start_index)
            logging.info(f"从进度恢复，开始索引: {start_index}")
        else:
            start_index = args.start_index
            progress = {
                "task": task_name,
                "processed_indices": [],
                "current_index": start_index
            }
            # ✅ 修复：立即保存初始进度
            self._save_progress(progress_file, progress)
        
        end_index = args.end_index if args.end_index != -1 else len(data) - 1
        
        logging.info(f"处理范围: 索引 {start_index} 到 {end_index}")
        
        # ✅ 修复：检查数据是否为空
        if not data:
            logging.warning("输入数据为空，跳过处理")
            return
        
        # 初始化输出数据
        if os.path.exists(output_json_path):
            try:
                with open(output_json_path, 'r', encoding='utf-8') as f:
                    output_data = json.load(f)
                logging.info(f"加载现有输出文件: {output_json_path}")
            except Exception as e:
                logging.warning(f"读取输出文件失败，创建新文件: {e}")
                output_data = []
        else:
            output_data = []
        
        # 确保输出数据长度与输入一致
        while len(output_data) < len(data):
            output_data.append(None)
        
        # ✅ 修复：在处理循环开始前保存进度
        self._save_progress(progress_file, progress)
        
        # 逐个文章处理
        for idx in range(start_index, end_index + 1):
            if idx >= len(data):
                logging.warning(f"索引 {idx} 超出数据范围，跳过")
                continue
                
            article = data[idx]
            article_name = article.get('name', f'文章_{idx}')
            article_url = article.get('metadata', {}).get('url', '未知URL')
            
            logging.info(f"\n{'='*60}")
            logging.info(f"处理第 {idx} 篇文章: {article_name}")
            logging.info(f"URL: {article_url}")
            logging.info(f"{'='*60}")
            
            # 更新进度
            progress["current_index"] = idx
            if idx not in progress["processed_indices"]:
                progress["processed_indices"].append(idx)
            
            # ✅ 修复：立即保存进度，无论处理结果如何
            self._save_progress(progress_file, progress)
            
            # 收集当前文章的实体和关系
            entity_a_terms = set()
            relation_terms = set()
            entity_context_parts = []
            relation_context_parts = []
            
            # 使用文章全文内容作为上下文
            article_content = article.get('content', '')
            
            # 遍历当前文章的所有结果
            for result in article.get('results', []):
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
                            head_label = labels[0]
                            tail_label = labels[1]
                            
                            # 收集label='a'的实体
                            if head_label and str(head_label).lower() == 'a' and head:
                                entity_a_terms.add(head)
                                entity_context_parts.append(f"{head} {relation} {tail}")
                            
                            if tail_label and str(tail_label).lower() == 'a' and tail:
                                entity_a_terms.add(tail)
                                entity_context_parts.append(f"{head} {relation} {tail}")
                            
                            # 收集所有关系词
                            if relation:
                                relation_terms.add(relation)
                            
                            relation_context_parts.append(f"{head} {relation} {tail}")
            
            # 转换为列表
            entity_a_list = sorted(list(entity_a_terms))
            relation_list = sorted(list(relation_terms))
            
            logging.info(f"收集到 {len(entity_a_list)} 个label='a'的实体")
            logging.info(f"收集到 {len(relation_list)} 个关系词")
            logging.info(f"实体: {entity_a_list}")
            logging.info(f"关系: {relation_list}")
            
            if not entity_a_list and not relation_list:
                logging.info(f"文章 {idx} 没有需要消歧的实体和关系，跳过")
                # 保存原始数据到输出
                output_data[idx] = article
                
                # 增量保存输出文件
                try:
                    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
                    current_output = [item if item is not None else data[i] for i, item in enumerate(output_data)]
                    with open(output_json_path, 'w', encoding='utf-8') as f:
                        json.dump(current_output, f, ensure_ascii=False, indent=2)
                    logging.info(f"增量保存: {output_json_path}")
                except Exception as e:
                    logging.error(f"保存输出文件失败: {e}")
                
                continue
            
            # 使用文章全文作为主要上下文，三元组作为补充
            entity_shared_context = article_content + " " + " ".join(entity_context_parts[:50])
            relation_shared_context = article_content + " " + " ".join(relation_context_parts[:50])
            
            # 初始化消歧器并执行消歧
            disambiguator = TermDisambiguator(api_provider="qianwen")
            
            try:
                updated_entities, updated_relations = disambiguator.Disambiguate(
                    entity_terms=entity_a_list,
                    relation_terms=relation_list,
                    entity_shared_context=entity_shared_context,
                    relation_shared_context=relation_shared_context
                )
                
                logging.info(f"消歧完成 - 更新后实体数: {len(updated_entities)}, 关系数: {len(updated_relations)}")
                
            except Exception as e:
                logging.error(f"文章 {idx} 消歧过程出错: {e}")
                import traceback
                logging.error(traceback.format_exc())
                # 出错时保存原始数据
                output_data[idx] = article
                
                # 增量保存输出文件
                try:
                    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
                    current_output = [item if item is not None else data[i] for i, item in enumerate(output_data)]
                    with open(output_json_path, 'w', encoding='utf-8') as f:
                        json.dump(current_output, f, ensure_ascii=False, indent=2)
                    logging.info(f"增量保存: {output_json_path}")
                except Exception as e:
                    logging.error(f"保存输出文件失败: {e}")
                
                continue
            
            # 创建映射字典
            entity_mapping = dict(zip(entity_a_list, updated_entities))
            relation_mapping = dict(zip(relation_list, updated_relations))
            
            logging.info(f"实体映射: {entity_mapping}")
            logging.info(f"关系映射: {relation_mapping}")
            
            # 复制文章数据并更新三元组
            updated_article = json.loads(json.dumps(article))  # 深拷贝
            
            # 更新当前文章的三元组
            updated_count = 0
            for result in updated_article.get('results', []):
                if 'output' not in result or 'relations' not in result['output']:
                    continue
                
                relations = result['output']['relations']
                for rel in relations:
                    if not isinstance(rel, dict) or 'triple' not in rel:
                        continue
                    
                    triple = rel['triple']
                    if len(triple) >= 3:
                        head, relation, tail = triple[0], triple[1], triple[2]
                        
                        # 更新术语
                        if head in entity_mapping and entity_mapping[head] != head:
                            rel['triple'][0] = entity_mapping[head]
                            updated_count += 1
                        
                        if relation in relation_mapping and relation_mapping[relation] != relation:
                            rel['triple'][1] = relation_mapping[relation]
                            updated_count += 1
                        
                        if tail in entity_mapping and entity_mapping[tail] != tail:
                            rel['triple'][2] = entity_mapping[tail]
                            updated_count += 1
            
            # 保存更新后的文章
            output_data[idx] = updated_article
            logging.info(f"文章 {idx} 更新完成，共更新 {updated_count} 个术语")
            
            # 增量保存输出文件
            try:
                os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
                current_output = [item if item is not None else data[i] for i, item in enumerate(output_data)]
                with open(output_json_path, 'w', encoding='utf-8') as f:
                    json.dump(current_output, f, ensure_ascii=False, indent=2)
                logging.info(f"增量保存: {output_json_path}")
            except Exception as e:
                logging.error(f"保存输出文件失败: {e}")
        
        logging.info("实体消歧任务完成")
        logging.info(f"进度文件: {progress_file}")
        logging.info(f"输出文件: {output_json_path}")
    
    def _run_concept_clustering(self, args):
        """
        执行局部概念聚类任务 - 按文章逐个处理
        
        Args:
            args: 命令行参数
        """
        logging.info("开始局部概念聚类任务（按文章逐个处理）")
        
        # 定义输入路径
        input_json_path = r"E:\KKGG\output\KG\输出.json"
        progress_file = args.progress_file
        task_name = "concept_clustering"
        
        # ✅ 修复：检查输入文件是否存在
        if not os.path.exists(input_json_path):
            logging.error(f"输入文件不存在: {input_json_path}")
            # 创建初始进度并保存
            progress = {
                "task": task_name,
                "processed_indices": [],
                "current_index": 0
            }
            self._save_progress(progress_file, progress)
            return
        
        # 读取JSON数据
        try:
            with open(input_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logging.info(f"成功读取输入文件: {input_json_path}, 共 {len(data)} 篇文章")
        except Exception as e:
            logging.error(f"读取输入文件失败: {e}")
            # ✅ 修复：即使读取失败也保存进度
            progress = {
                "task": task_name,
                "processed_indices": [],
                "current_index": 0
            }
            self._save_progress(progress_file, progress)
            return
        
        # 加载进度
        progress = self._load_progress(progress_file, task_name)
        if args.resume:
            start_index = progress.get("current_index", args.start_index)
            logging.info(f"从进度恢复，开始索引: {start_index}")
        else:
            start_index = args.start_index
            progress = {
                "task": task_name,
                "processed_indices": [],
                "current_index": start_index
            }
            # ✅ 修复：立即保存初始进度
            self._save_progress(progress_file, progress)
        
        end_index = args.end_index if args.end_index != -1 else len(data) - 1
        
        logging.info(f"处理范围: 索引 {start_index} 到 {end_index}")
        
        # ✅ 修复：检查数据是否为空
        if not data:
            logging.warning("输入数据为空，跳过处理")
            return
        
        # ✅ 修复：在处理循环开始前保存进度
        self._save_progress(progress_file, progress)
        
        # 逐个文章处理
        for idx in range(start_index, end_index + 1):
            if idx >= len(data):
                logging.warning(f"索引 {idx} 超出数据范围，跳过")
                continue
                
            article = data[idx]
            article_name = article.get('name', f'文章_{idx}')
            article_url = article.get('metadata', {}).get('url', '未知URL')
            
            logging.info(f"\n{'='*60}")
            logging.info(f"处理第 {idx} 篇文章: {article_name}")
            logging.info(f"URL: {article_url}")
            logging.info(f"{'='*60}")
            
            # 更新进度
            progress["current_index"] = idx
            if idx not in progress["processed_indices"]:
                progress["processed_indices"].append(idx)
            
            # ✅ 修复：立即保存进度，无论处理结果如何
            self._save_progress(progress_file, progress)
            
            # 收集当前文章的label='b'的实体
            entity_b_terms = set()
            context_parts = []
            
            # 使用文章全文内容作为上下文
            article_content = article.get('content', '')
            
            # 遍历当前文章的所有结果
            for result in article.get('results', []):
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
                            head_label = labels[0]
                            tail_label = labels[1]
                            
                            # 收集label='b'的实体
                            if head_label and str(head_label).lower() == 'b' and head:
                                entity_b_terms.add(head)
                                context_parts.append(f"{head} {relation} {tail}")
                            
                            if tail_label and str(tail_label).lower() == 'b' and tail:
                                entity_b_terms.add(tail)
                                context_parts.append(f"{head} {relation} {tail}")
            
            # 转换为列表
            entity_b_list = sorted(list(entity_b_terms))
            
            logging.info(f"收集到 {len(entity_b_list)} 个label='b'的实体")
            logging.info(f"实体: {entity_b_list}")
            
            if not entity_b_list:
                logging.info(f"文章 {idx} 没有需要聚类的实体，跳过")
                continue
            
            # 使用文章全文作为主要上下文，三元组作为补充
            shared_context = article_content + " " + " ".join(context_parts[:100])
            
            # 初始化聚类器并执行聚类
            clusterer = TermDisambiguator(api_provider="qianwen")
            
            try:
                cluster_result = clusterer.clusterer(
                    terms=entity_b_list,
                    shared_context=shared_context
                )
                
                logging.info(f"文章 {idx} 聚类完成")
                logging.info(f"聚类结果包含 {len(cluster_result)} 个术语的三元组信息")
                
                # 记录部分聚类结果
                for i, (term, triples) in enumerate(list(cluster_result.items())[:3]):
                    logging.info(f"术语 '{term}' 的三元组数量: {len(triples)}")
                    
            except Exception as e:
                logging.error(f"文章 {idx} 聚类过程出错: {e}")
                import traceback
                logging.error(traceback.format_exc())
                continue
        
        logging.info("局部概念聚类任务完成")
        logging.info(f"进度文件: {progress_file}")


if __name__ == "__main__":
    cli = MainCLI()
    cli.execute()
