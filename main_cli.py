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
import re
from pathlib import Path
from typing import List, Dict, Set, Tuple
from cli import CLI
from entity_extraction.extractor import EntityExtractor
from relation_extraction.extractor import RelationExtractor
from disambiguator_clusterer import TermDisambiguator
from Quality_evaluation.extractor import EvaluationExtractor


class MainCLI(CLI):
    """图谱构建主命令行工具，提供实体提取、关系提取、实体消歧、局部概念聚类等功能"""
    
    def _add_arguments(self):
        """添加命令行参数"""
        super()._add_arguments()
        
        # 添加任务选择参数
        self.parser.add_argument('--task', type=str, required=True,
                                choices=['entity_extraction', 'relation_extraction', 
                                         'entity_evaluation', 'relation_evaluation',
                                         'entity_disambiguation', 'concept_clustering', 'all'],
                                help='要执行的任务')
        
        # 实体提取相关参数 
        self.parser.add_argument('--entity_model', type=str, default='default',
                                help='实体提取使用的模型')
        self.parser.add_argument('--entity_input_dir', type=str, required=False,
                                help='实体提取的输入目录）')
        self.parser.add_argument('--entity_output_dir', type=str, required=False,
                                help='实体提取的输出目录')
        self.parser.add_argument('--entity_error_dir', type=str, default=None,
                                 help='实体提取失败条目保存目录（默认: entity_output_dir/errors）')
        self.parser.add_argument('--entity_prompt', type=str, required=False,
                                help='实体提取的 Prompt 模板路径')
        self.parser.add_argument('--entity_threshold', type=float, default=0.5,
                                help='实体提取的置信度阈值')
        
        #  关系提取相关参数 
        self.parser.add_argument('--relation_model', type=str, default='default',
                                help='关系提取使用的模型')
        self.parser.add_argument('--relation_input_dir', type=str, required=False,
                                help='关系提取的输入目录')
        self.parser.add_argument('--relation_output_dir', type=str, required=False,
                                help='关系提取的输出目录')
        self.parser.add_argument('--relation_error_dir', type=str, default=None,
                                 help='关系提取失败条目保存目录（默认: relation_output_dir/errors）')
        self.parser.add_argument('--relation_prompt', type=str, required=False,
                                help='关系提取的 Prompt 模板路径')
        self.parser.add_argument('--relation_threshold', type=float, default=0.5,
                                help='关系提取的置信度阈值')
        
        # === 實體評估新增參數 ===
        self.parser.add_argument('--entity_eval_model1', type=str, default='default',
                                help='實體評估奇數輪使用的模型')
        self.parser.add_argument('--entity_eval_model2', type=str, default='default',
                                help='實體評估偶數輪使用的模型')
        self.parser.add_argument('--entity_eval_prompt', type=str, default='prompt/evaluation_entities',
                                help='實體評估 Prompt 路徑')
        self.parser.add_argument('--entity_eval_input_dir', type=str, required=False,
                                help='實體評估輸入目錄（待評估的實體提取結果）')
        self.parser.add_argument('--entity_eval_output_dir', type=str, required=False,
                                help='實體評估輸出目錄')
        self.parser.add_argument('--entity_max_iterations', type=int, default=5,
                                help='實體最大迭代修正次數')

        # === 關係評估新增參數 ===
        self.parser.add_argument('--relation_eval_model1', type=str, default='default',
                                help='關係評估奇數輪使用的模型')
        self.parser.add_argument('--relation_eval_model2', type=str, default='default',
                                help='關係評估偶數輪使用的模型')
        self.parser.add_argument('--relation_eval_prompt', type=str, default='prompt/evaluation_relations',
                                help='關係評估 Prompt 路徑')
        self.parser.add_argument('--relation_eval_input_dir', type=str, required=False,
                                help='關係評估輸入目錄（待評估的關係提取結果）')
        self.parser.add_argument('--relation_eval_output_dir', type=str, required=False,
                                help='關係評估輸出目錄')
        self.parser.add_argument('--relation_max_iterations', type=int, default=5,
                                help='關係最大迭代修正次數')
        
        # === 評估過程記錄目錄 ===
        self.parser.add_argument('--record_dir', type=str, default="Output/record",
                                help='評估過程詳細記錄目錄（包含每輪LLM輸出、修正過程），默認: Output/record')
        self.parser.add_argument('--no_record', action='store_true',
                                help='完全禁用過程記錄（節省磁碟空間，僅保留最終結果）')

        # 实体消歧相关参数
        self.parser.add_argument('--disambiguation_method', type=str, default='default',
                                help='实体消歧使用的方法')
        self.parser.add_argument('--disambiguation_input_dir', type=str,
                                default=r'E:\KKGG\output\KG\test',
                                help='实体消歧的输入目录')
        self.parser.add_argument('--disambiguation_output_dir', type=str,
                                default=r'E:\KKGG\output\KG\test_削岐后',
                                help='实体消歧的输出目录')
        self.parser.add_argument('--kb_path', type=str, default='',
                                help='知识库路径，用于实体消歧')
        
        # 概念聚类相关参数
        self.parser.add_argument('--clustering_method', type=str, default='default',
                                help='概念聚类使用的方法')
        self.parser.add_argument('--clustering_input_dir', type=str,
                                default=r'E:\KKGG\output\KG\test',
                                help='概念聚类的输入目录')
        self.parser.add_argument('--cluster_output_file', type=str,
                                default=r'E:\KKGG\output\terms\test_entity_cluster_triples.json',
                                help='聚类输出文件路径')
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
        self.parser.add_argument('--terms_base_dir', type=str,
                                default=r'E:\KKGG\output\terms',
                                help='术语库基础目录路径（默认可用）')
        self.parser.add_argument('--no_pid', action='store_true',
                                help='禁用PID命名，使用默认文件名')    
        
        # 新增：日志目录参数
        self.parser.add_argument('--log_dir', type=str, default='./logs',
                                help='日志输出目录（默认: ./logs）')
        
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
            self._run_entity_evaluation(args)
            self._run_relation_evaluation(args)
            self._run_entity_disambiguation(args)
            self._run_concept_clustering(args)
        elif args.task == 'entity_extraction':
            self._run_entity_extraction(args)
        elif args.task == 'relation_extraction':
            self._run_relation_extraction(args)
        elif args.task == 'entity_evaluation':
            self._run_entity_evaluation(args)
        elif args.task == 'relation_evaluation':
            self._run_relation_evaluation(args)
        elif args.task == 'entity_disambiguation':
            self._run_entity_disambiguation(args)
        elif args.task == 'concept_clustering':
            self._run_concept_clustering(args)
        
        logging.info(f"任务 {args.task} 执行完成")
    
    def _extract_file_index(self, filename):
        """
        从文件名中提取索引数字
        
        Args:
            filename: 文件名
            
        Returns:
            int: 文件索引，如果无法提取则返回-1
        """
        # 匹配文件名开头的数字部分，例如 "00003_中国大陆银行列表.json" -> 3
        match = re.match(r'^(\d+)', filename.stem)
        if match:
            return int(match.group(1))
        return -1
    
    def _get_sorted_files(self, input_dir):
        """
        获取按索引排序的文件列表
        
        Args:
            input_dir: 输入目录
            
        Returns:
            list: 按索引排序的文件路径列表
        """
        input_files = sorted(input_dir.glob("*.json")) + sorted(input_dir.glob("*.jsonl"))
        
        # 提取文件索引并排序
        files_with_index = []
        for file_path in input_files:
            index = self._extract_file_index(file_path)
            files_with_index.append((index, file_path))
        
        # 按索引排序
        files_with_index.sort(key=lambda x: x[0])
        
        # 返回排序后的文件路径
        return [file_path for _, file_path in files_with_index]
    
    def _run_entity_extraction(self, args):
        logging.info("开始实体提取任务（.json 输入 → .jsonl 输出，条目级容错）")

        if not args.entity_input_dir or not args.entity_output_dir:
            raise ValueError("--entity_input_dir 和 --entity_output_dir 必须指定")

        input_dir = Path(args.entity_input_dir)
        output_dir = Path(args.entity_output_dir)
        error_dir = Path(args.entity_error_dir) if args.entity_error_dir else output_dir / "errors"
        output_dir.mkdir(parents=True, exist_ok=True)

        extractor = EntityExtractor(
            model_name=args.entity_model,
            threshold=args.entity_threshold
        )

        input_files = self._get_sorted_files(input_dir)
        if not input_files:
            logging.warning(f"实体输入目录中无 .json 文件: {input_dir}")
            return

        total_processed = total_success = total_failure = 0

        for idx, input_file in enumerate(input_files):
            if idx < args.start_index:
                continue
            if args.end_index >= 0 and idx > args.end_index:
                break

            output_file = output_dir / (input_file.stem + ".jsonl")
            error_file = error_dir / (input_file.stem + ".errors.jsonl")

            if args.resume and output_file.exists():
                logging.info(f"跳过（已存在）: {output_file.name}")
                continue

            logging.info(f"处理实体文件: {input_file.name}")

            try:
                with open(input_file, 'r', encoding='utf-8') as f:
                    records = json.load(f)
                if not isinstance(records, list):
                    raise ValueError("JSON 文件根元素必须是数组")
            except Exception as e:
                logging.error(f"加载失败 {input_file}: {e}")
                continue

            success_count = failure_count = 0
            error_buffer = []

            with open(output_file, 'w', encoding='utf-8') as out_f:
                for record_idx, record in enumerate(records):
                    try:
                        result = extractor.extract_single(record, prompt_path=args.entity_prompt)
                        out_f.write(json.dumps(result, ensure_ascii=False) + '\n')
                        success_count += 1
                    except Exception as e:
                        failure_count += 1
                        error_buffer.append({
                            "source_file": str(input_file),
                            "record_index": record_idx,
                            "record": record,
                            "error": str(e)
                        })

            total_processed += len(records)
            total_success += success_count
            total_failure += failure_count

            if error_buffer:
                error_dir.mkdir(parents=True, exist_ok=True)
                with open(error_file, 'w', encoding='utf-8') as err_f:
                    for err in error_buffer:
                        err_f.write(json.dumps(err, ensure_ascii=False) + '\n')
                logging.warning(f"  → {failure_count} 条失败，错误日志: {error_file}")
            else:
                logging.info(f"  → 全部 {success_count} 条成功")

        logging.info(f"实体提取完成：共 {total_processed} 条，成功 {total_success}，失败 {total_failure}")

    def _run_relation_extraction(self, args):
        logging.info("开始关系提取任务（.jsonl 输入/输出，条目级容错）")

        if not args.relation_input_dir or not args.relation_output_dir:
            raise ValueError("--relation_input_dir 和 --relation_output_dir 必须指定")

        input_dir = Path(args.relation_input_dir)
        output_dir = Path(args.relation_output_dir)
        error_dir = Path(args.relation_error_dir) if args.relation_error_dir else output_dir / "errors"
        output_dir.mkdir(parents=True, exist_ok=True)

        extractor = RelationExtractor(
            model_name=args.relation_model,
            threshold=args.relation_threshold
        )

        input_files = self._get_sorted_files(input_dir)
        if not input_files:
            logging.warning(f"关系输入目录中无 .jsonl 文件: {input_dir}")
            return

        total_processed = total_success = total_failure = 0

        for idx, input_file in enumerate(input_files):
            if idx < args.start_index:
                continue
            if args.end_index >= 0 and idx > args.end_index:
                break

            output_file = output_dir / (input_file.stem + ".jsonl")
            error_file = error_dir / (input_file.stem + ".errors.jsonl")

            if args.resume and output_file.exists():
                logging.info(f"跳过（已存在）: {output_file.name}")
                continue

            logging.info(f"处理关系文件: {input_file.name}")

            records = []
            with open(input_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except Exception as e:
                            logging.warning(f"跳过无效 JSON 行 {line_num} in {input_file}: {e}")

            success_count = failure_count = 0
            error_buffer = []

            with open(output_file, 'w', encoding='utf-8') as out_f:
                for record_idx, record in enumerate(records):
                    try:
                        result = extractor.extract_single(record, prompt_path=args.relation_prompt)
                        out_f.write(json.dumps(result, ensure_ascii=False) + '\n')
                        success_count += 1
                    except Exception as e:
                        failure_count += 1
                        error_buffer.append({
                            "source_file": str(input_file),
                            "record_index": record_idx,
                            "record": record,
                            "error": str(e)
                        })

            total_processed += len(records)
            total_success += success_count
            total_failure += failure_count

            if error_buffer:
                error_dir.mkdir(parents=True, exist_ok=True)
                with open(error_file, 'w', encoding='utf-8') as err_f:
                    for err in error_buffer:
                        err_f.write(json.dumps(err, ensure_ascii=False) + '\n')
                logging.warning(f"  → {failure_count} 条失败，错误日志: {error_file}")
            else:
                logging.info(f"  → 全部 {success_count} 条成功")

        logging.info(f"关系提取完成：共 {total_processed} 条，成功 {total_success}，失败 {total_failure}")
    
    def _run_entity_evaluation(self, args):
        logging.info("開始實體評估任務（多輪迭代自我修正）")
        
        input_dir = Path(args.entity_eval_input_dir or args.entity_output_dir)
        output_dir = Path(args.entity_eval_output_dir or "output/entities_evaluated")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not input_dir.exists():
            logging.error(f"輸入目錄不存在: {input_dir}")
            return
        
        record_base = None if args.no_record else args.record_dir
        
        evaluator = EvaluationExtractor(
            eval_model1=args.entity_eval_model1,
            eval_model2=args.entity_eval_model2,
            eval_prompt_path=args.entity_eval_prompt,
            max_iterations=args.entity_max_iterations,
            record_base_dir=record_base
        )
        
        # 直接使用你已有的排序方法
        input_files = self._get_sorted_files(input_dir)
        if not input_files:
            logging.warning(f"輸入目錄中無 JSON 文件: {input_dir}")
            return
        
        logging.info(f"發現 {len(input_files)} 個待評估文件")
        
        for idx, input_file in enumerate(input_files):
            output_file = output_dir / input_file.name
            
            # 簡單判斷：輸出文件已存在 = 跳過（你不需要斷點續傳）
            if output_file.exists():
                logging.info(f"跳過（已存在）: {output_file.name}")
                continue
                
            logging.info(f"\n{'='*80}")
            logging.info(f"[{idx+1}/{len(input_files)}] 正在評估實體: {input_file.name}")
            logging.info(f"{'='*80}")
            
            try:
                result = evaluator.evaluate_and_correct(
                    input_file=input_file,
                    output_file=output_file,
                    record_dir=Path(args.record_dir) / input_file.stem if not args.no_record else None,          # 這裡是你想要的 record 根目錄
                    task_type="entity",                         # 或 "relation"
                    chunk_id=input_file.stem
                )
                logging.info(f"實體評估完成 → {output_file.name} （共 {result['total_iterations']} 輪）")
                
            except Exception as e:
                logging.error(f"實體評估失敗 {input_file.name}: {e}")
                import traceback
                logging.error(traceback.format_exc())
        
        logging.info("實體評估任務全部完成！")

    def _run_relation_evaluation(self, args):
        logging.info("開始關係評估任務（多輪迭代自我修正 + 保留最終實體）")
        
        input_dir = Path(args.relation_eval_input_dir or args.relation_output_dir)
        output_dir = Path(args.relation_eval_output_dir or "output/relations_evaluated")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not input_dir.exists():
            logging.error(f"輸入目錄不存在: {input_dir}")
            return
            
        record_base = None if args.no_record else args.record_dir
        
        evaluator = EvaluationExtractor(
            eval_model1=args.relation_eval_model1,
            eval_model2=args.relation_eval_model2,
            eval_prompt_path=args.relation_eval_prompt,
            max_iterations=args.relation_max_iterations,
            record_base_dir=record_base
        )
        
        input_files = self._get_sorted_files(input_dir)
        if not input_files:
            logging.warning(f"輸入目錄中無 JSON 文件: {input_dir}")
            return
        
        logging.info(f"發現 {len(input_files)} 個待評估文件")
        
        for idx, input_file in enumerate(input_files):
            output_file = output_dir / input_file.name
            
            if output_file.exists():
                logging.info(f"跳過（已存在）: {output_file.name}")
                continue
                
            logging.info(f"\n{'='*80}")
            logging.info(f"[{idx+1}/{len(input_files)}] 正在評估關係: {input_file.name}")
            logging.info(f"{'='*80}")
            
            try:
                result = evaluator.evaluate_and_correct(
                    input_file=input_file,
                    output_file=output_file,
                    record_dir=Path(args.record_dir) / input_file.stem if not args.no_record else None,
                    task_type="relation",                         # 或 "relation"
                    chunk_id=input_file.stem
                )
                logging.info(f"關係評估完成 → {output_file.name} （共 {result['total_iterations']} 輪）")
                
            except Exception as e:
                logging.error(f"關係評估失敗 {input_file.name}: {e}")
                import traceback
                logging.error(traceback.format_exc())
        
        logging.info("關係評估任務全部完成！")

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
        执行实体消歧任务 - 处理目录内的JSON文件
        
        Args:
            args: 命令行参数
        """
        logging.info("开始实体消歧任务（处理目录内文件）")
        
        # 定义输入输出路径
        input_dir = Path(args.disambiguation_input_dir)
        output_dir = Path(args.disambiguation_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        progress_file = args.progress_file
        task_name = "entity_disambiguation"
        
        # 确定是否使用PID
        pid = None if args.no_pid else os.getpid()
        
        # 创建消歧器时传递新参数（仅修改这一行）
        disambiguator = TermDisambiguator(
            api_provider="qianwen",
            base_terms_dir=args.terms_base_dir,
            process_id=pid
        )
        
        # 获取按索引排序的输入文件
        input_files = self._get_sorted_files(input_dir)
        if not input_files:
            logging.warning(f"输入目录中没有找到JSON文件: {input_dir}")
            return
        
        logging.info(f"找到 {len(input_files)} 个输入文件")
        
        # 加载进度
        progress = self._load_progress(progress_file, task_name)
        
        # 计算实际开始和结束索引
        if args.resume:
            # 从进度恢复，找到第一个未处理的文件
            processed_files = set(progress.get("processed_files", []))
            start_index = 0
            # 找到第一个未处理的文件
            for i, file_path in enumerate(input_files):
                if file_path.name not in processed_files:
                    start_index = i
                    break
            else:
                # 所有文件都已处理
                start_index = len(input_files)
            logging.info(f"从进度恢复，开始索引: {start_index}")
        else:
            # 从头开始，找到第一个符合start_index的文件
            start_index = 0
            for i, file_path in enumerate(input_files):
                file_index = self._extract_file_index(file_path)
                if file_index >= args.start_index:
                    start_index = i
                    break
        
        # 计算结束索引
        if args.end_index >= 0:
            end_index = -1
            for i, file_path in enumerate(input_files):
                file_index = self._extract_file_index(file_path)
                if file_index > args.end_index:
                    end_index = i - 1
                    break
            else:
                end_index = len(input_files) - 1
        else:
            end_index = len(input_files) - 1
        
        # 初始化进度
        progress.update({
            "task": task_name,
            "total_files": len(input_files),
            "current_index": start_index
        })
        self._save_progress(progress_file, progress)
        
        logging.info(f"处理范围: 文件索引 {start_index} 到 {end_index} (共 {end_index - start_index + 1} 个文件)")
        
        # 逐个文件处理
        processed_count = 0
        for idx in range(start_index, end_index + 1):
            if idx >= len(input_files):
                logging.warning(f"索引 {idx} 超出文件范围，跳过")
                continue
                
            input_file = input_files[idx]
            output_file = output_dir / input_file.name
            file_index = self._extract_file_index(input_file)
            
            # 检查文件是否已处理
            if args.resume and input_file.name in progress.get("processed_files", []):
                logging.info(f"跳过（已处理）: {input_file.name}")
                continue
            
            logging.info(f"\n{'='*60}")
            logging.info(f"处理第 {idx}/{len(input_files)} 个文件 (索引{file_index}): {input_file.name}")
            logging.info(f"输出文件: {output_file.name}")
            logging.info(f"{'='*60}")
            
            # 处理单个文件
            try:
                # 读取输入文件
                with open(input_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                logging.info(f"成功读取文件: {input_file}, 共 {len(data)} 篇文章")
                
                # 初始化输出数据（与输入结构相同）
                output_data = []
                
                # 逐个文章处理
                for article_idx, article in enumerate(data):
                    article_name = article.get('name', f'文章_{article_idx}')
                    
                    logging.info(f"处理文章 {article_idx}: {article_name}")
                    
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
                    
                    if not entity_a_list and not relation_list:
                        logging.info(f"文章 {article_idx} 没有需要消歧的实体和关系，保持原样")
                        output_data.append(article)
                        continue
                    
                    # 使用文章全文作为主要上下文，三元组作为补充
                    entity_shared_context = article_content + " " + " ".join(entity_context_parts[:50])
                    relation_shared_context = article_content + " " + " ".join(relation_context_parts[:50])
                    
                    
                    try:
                        updated_entities, updated_relations = disambiguator.Disambiguate(
                            entity_terms=entity_a_list,
                            relation_terms=relation_list,
                            entity_shared_context=entity_shared_context,
                            relation_shared_context=relation_shared_context
                        )
                        
                        logging.info(f"消歧完成 - 更新后实体数: {len(updated_entities)}, 关系数: {len(updated_relations)}")
                        
                    except Exception as e:
                        logging.error(f"文章 {article_idx} 消歧过程出错: {e}")
                        # 出错时保存原始数据
                        output_data.append(article)
                        continue
                    
                    # 创建映射字典
                    entity_mapping = dict(zip(entity_a_list, updated_entities))
                    relation_mapping = dict(zip(relation_list, updated_relations))
                    
                    logging.info(f"实体映射: {len(entity_mapping)} 个")
                    logging.info(f"关系映射: {len(relation_mapping)} 个")
                    
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
                    output_data.append(updated_article)
                    logging.info(f"文章 {article_idx} 更新完成，共更新 {updated_count} 个术语")
                
                # 保存输出文件
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)
                logging.info(f"输出文件保存: {output_file}")
                
                # 更新进度
                processed_files = progress.get("processed_files", [])
                if input_file.name not in processed_files:
                    processed_files.append(input_file.name)
                progress.update({
                    "processed_files": processed_files,
                    "current_index": idx + 1,  # 下一个要处理的文件索引
                    "processed_count": len(processed_files)
                })
                self._save_progress(progress_file, progress)
                
                processed_count += 1
                logging.info(f"文件处理完成: {input_file.name} (进度: {idx + 1}/{len(input_files)})")
                
            except Exception as e:
                logging.error(f"处理文件失败 {input_file.name}: {e}")
                import traceback
                logging.error(traceback.format_exc())
                continue
        
        logging.info(f"实体消歧任务完成，共处理 {processed_count} 个文件")
        logging.info(f"进度文件: {progress_file}")
        logging.info(f"输出目录: {output_dir}")
    def _run_concept_clustering(self, args):
        """
        执行局部概念聚类任务 - 处理目录内的JSON文件
        
        Args:
            args: 命令行参数
        """
        logging.info("开始局部概念聚类任务（处理目录内文件）")
        
        # 定义输入路径
        input_dir = Path(args.clustering_input_dir)
        cluster_output_file = Path(args.cluster_output_file)
        progress_file = args.progress_file
        task_name = "concept_clustering"
        
        # 确定是否使用PID
        pid = None if args.no_pid else os.getpid()
        
        # 创建聚类器时传递新参数（仅修改这一行）
        clusterer = TermDisambiguator(
            api_provider="qianwen",
            base_terms_dir=args.terms_base_dir,
            process_id=pid
        )
        
        # 获取按索引排序的输入文件
        input_files = self._get_sorted_files(input_dir)
        if not input_files:
            logging.warning(f"输入目录中没有找到JSON文件: {input_dir}")
            return
        
        logging.info(f"找到 {len(input_files)} 个输入文件")
        
        # 加载进度
        progress = self._load_progress(progress_file, task_name)
        
        # 计算实际开始和结束索引
        if args.resume:
            # 从进度恢复，找到第一个未处理的文件
            processed_files = set(progress.get("processed_files", []))
            start_index = 0
            # 找到第一个未处理的文件
            for i, file_path in enumerate(input_files):
                if file_path.name not in processed_files:
                    start_index = i
                    break
            else:
                # 所有文件都已处理
                start_index = len(input_files)
            logging.info(f"从进度恢复，开始索引: {start_index}")
        else:
            # 从头开始，找到第一个符合start_index的文件
            start_index = 0
            for i, file_path in enumerate(input_files):
                file_index = self._extract_file_index(file_path)
                if file_index >= args.start_index:
                    start_index = i
                    break
        
        # 计算结束索引
        if args.end_index >= 0:
            end_index = -1
            for i, file_path in enumerate(input_files):
                file_index = self._extract_file_index(file_path)
                if file_index > args.end_index:
                    end_index = i - 1
                    break
            else:
                end_index = len(input_files) - 1
        else:
            end_index = len(input_files) - 1
        
        # 初始化进度
        progress.update({
            "task": task_name,
            "total_files": len(input_files),
            "current_index": start_index
        })
        self._save_progress(progress_file, progress)
        
        logging.info(f"处理范围: 文件索引 {start_index} 到 {end_index} (共 {end_index - start_index + 1} 个文件)")
        
        # 收集所有文件的label='b'的实体
        all_entity_b_terms = set()
        all_context_parts = []
        processed_count = 0
        
        # 逐个文件处理，收集实体
        for idx in range(start_index, end_index + 1):
            if idx >= len(input_files):
                logging.warning(f"索引 {idx} 超出文件范围，跳过")
                continue
                
            input_file = input_files[idx]
            file_index = self._extract_file_index(input_file)
            
            # 检查文件是否已处理
            if args.resume and input_file.name in progress.get("processed_files", []):
                logging.info(f"跳过（已处理）: {input_file.name}")
                continue
            
            logging.info(f"\n{'='*60}")
            logging.info(f"处理第 {idx}/{len(input_files)} 个文件 (索引{file_index}): {input_file.name}")
            logging.info(f"{'='*60}")
            
            # 从文件中收集实体
            try:
                # 读取文件
                with open(input_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                entity_b_terms = set()
                context_parts = []
                
                # 遍历所有文章
                for article in data:
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
                
                all_entity_b_terms.update(entity_b_terms)
                all_context_parts.extend(context_parts)
                
                # 更新进度
                processed_files = progress.get("processed_files", [])
                if input_file.name not in processed_files:
                    processed_files.append(input_file.name)
                progress.update({
                    "processed_files": processed_files,
                    "current_index": idx + 1,  # 下一个要处理的文件索引
                    "processed_count": len(processed_files)
                })
                self._save_progress(progress_file, progress)
                
                processed_count += 1
                logging.info(f"文件处理完成: {input_file.name}, 收集到 {len(entity_b_terms)} 个实体 (进度: {idx + 1}/{len(input_files)})")
                
            except Exception as e:
                logging.error(f"处理文件失败 {input_file.name}: {e}")
                import traceback
                logging.error(traceback.format_exc())
                continue
        
        # 执行概念聚类
        if all_entity_b_terms:
            entity_b_list = sorted(list(all_entity_b_terms))
            logging.info(f"总共收集到 {len(entity_b_list)} 个label='b'的实体")
            
            # 合并所有上下文
            shared_context = " ".join(all_context_parts[:200])  # 限制上下文长度

            
            try:
                cluster_result = clusterer.clusterer(
                    terms=entity_b_list,
                    shared_context=shared_context
                )
                
                logging.info(f"概念聚类完成")
                logging.info(f"聚类结果包含 {len(cluster_result)} 个术语的三元组信息")
                logging.info(f"输出文件: {cluster_output_file}")
                
            except Exception as e:
                logging.error(f"聚类过程出错: {e}")
                import traceback
                logging.error(traceback.format_exc())
        else:
            logging.info("没有收集到需要聚类的实体")
        
        logging.info(f"局部概念聚类任务完成，共处理 {processed_count} 个文件")
        logging.info(f"进度文件: {progress_file}")

if __name__ == "__main__":
    cli = MainCLI()
    cli.execute()
