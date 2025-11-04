#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
评估命令行工具，用于评估消歧后的实体和关系提取结果
"""

import argparse
import logging
import os
from project.cli import CLI


class EvalCLI(CLI):
    """评估工具，用于评估实体提取和关系提取的结果质量"""
    
    def _add_arguments(self):
        """添加命令行参数"""
        super()._add_arguments()
        
        # 添加评估任务选择参数
        self.parser.add_argument('--eval_task', type=str, required=True,
                                choices=['entity', 'relation', 'all'],
                                help='要评估的任务类型')
        
        # 评估数据相关参数
        self.parser.add_argument('--gold_data', type=str, required=True,
                                help='金标数据路径')
        self.parser.add_argument('--pred_data', type=str, required=True,
                                help='预测数据路径')
        
        # 评估指标相关参数
        self.parser.add_argument('--metrics', type=str, default='precision,recall,f1',
                                help='评估指标，用逗号分隔')
        self.parser.add_argument('--detailed', action='store_true',
                                help='是否输出详细评估结果')
    
    def run(self, args):
        """
        执行评估任务
        
        Args:
            args: 解析后的命令行参数
        """
        logging.info(f"开始评估任务: {args.eval_task}")
        
        if args.eval_task == 'all':
            self._evaluate_entity(args)
            self._evaluate_relation(args)
        elif args.eval_task == 'entity':
            self._evaluate_entity(args)
        elif args.eval_task == 'relation':
            self._evaluate_relation(args)
        
        logging.info(f"评估任务 {args.eval_task} 完成")
    
    def _evaluate_entity(self, args):
        """
        评估实体提取结果
        
        Args:
            args: 命令行参数
        """
        logging.info("开始评估实体提取结果")
        
        # 这里应该调用实体评估相关的代码
        # 示例代码框架，具体实现需要根据实际情况编写
        from project.entity_disambiguation.evaluator import EntityEvaluator
        
        evaluator = EntityEvaluator(
            metrics=args.metrics.split(','),
            detailed=args.detailed
        )
        
        results = evaluator.evaluate(
            gold_path=args.gold_data,
            pred_path=args.pred_data
        )
        
        # 输出评估结果
        for metric, value in results.items():
            logging.info(f"实体评估 - {metric}: {value}")
        
        # 保存详细结果
        if args.detailed:
            output_path = os.path.join(args.output_dir, 'entity_eval_results.json')
            evaluator.save_results(results, output_path)
            logging.info(f"详细评估结果已保存至: {output_path}")
        
        logging.info("实体评估完成")
    
    def _evaluate_relation(self, args):
        """
        评估关系提取结果
        
        Args:
            args: 命令行参数
        """
        logging.info("开始评估关系提取结果")
        
        # 这里应该调用关系评估相关的代码
        # 示例代码框架，具体实现需要根据实际情况编写
        from project.relation_extraction.evaluator import RelationEvaluator
        
        evaluator = RelationEvaluator(
            metrics=args.metrics.split(','),
            detailed=args.detailed
        )
        
        results = evaluator.evaluate(
            gold_path=args.gold_data,
            pred_path=args.pred_data
        )
        
        # 输出评估结果
        for metric, value in results.items():
            logging.info(f"关系评估 - {metric}: {value}")
        
        # 保存详细结果
        if args.detailed:
            output_path = os.path.join(args.output_dir, 'relation_eval_results.json')
            evaluator.save_results(results, output_path)
            logging.info(f"详细评估结果已保存至: {output_path}")
        
        logging.info("关系评估完成")


if __name__ == "__main__":
    cli = EvalCLI()
    cli.execute()