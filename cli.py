#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基础命令行接口类
提供命令行工具的基础功能和通用方法
"""

import argparse
import logging
import os
import sys
from abc import ABC, abstractmethod


class CLI(ABC):
    """
    命令行接口基类，提供基础功能和通用方法
    所有CLI工具都应该继承此类
    """
    
    def __init__(self):
        """初始化CLI基类，设置基本参数解析器"""
        self.parser = argparse.ArgumentParser(description=self.__doc__)
        self._add_arguments()
        
    def _add_arguments(self):
        """添加通用命令行参数"""
        self.parser.add_argument('--log_dir', type=str, default='../log', 
                                help='日志目录路径')
        self.parser.add_argument('--data_dir', type=str, default='../data',
                                help='数据目录路径')
        self.parser.add_argument('--output_dir', type=str, default='../output',
                                help='输出目录路径')
        self.parser.add_argument('--debug', action='store_true',
                                help='是否开启调试模式')
    
    @abstractmethod
    def run(self, args):
        """
        执行CLI命令
        
        Args:
            args: 解析后的命令行参数
        """
        pass
    
    def execute(self):
        """解析命令行参数并执行命令"""
        args = self.parser.parse_args()
        self._setup_logging(args)
        return self.run(args)
    
    def _setup_logging(self, args):
        """设置日志配置"""
        log_level = logging.DEBUG if args.debug else logging.INFO
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
            
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.FileHandler(os.path.join(args.log_dir, 'app.log')),
                logging.StreamHandler(sys.stdout)
            ]
        )