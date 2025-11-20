#!/bin/bash

# ===========================================
# 关系提取脚本
# 用法: ./extract_relations.sh <input_dir> <output_dir> <prompt_file> [model] [threshold]
# 示例: ./extract_relations.sh ../output ../output_rel ../prompts/relation_prompt.txt default 0.5
# ===========================================

set -e

if [ "$#" -lt 3 ]; then
    echo "错误：至少需要 3 个参数"
    echo "用法: $0 <input_dir> <output_dir> <prompt_file> [model] [threshold]"
    echo "  - input_dir: 输入 JSONL 文件目录（通常是实体提取的输出）"
    echo "  - output_dir: 输出 JSONL 目录"
    echo "  - prompt_file: Prompt 模板文件路径"
    echo "  - model (可选): 模型名，默认 'default'"
    echo "  - threshold (可选): 置信度阈值，默认 0.5"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"
PROMPT_FILE="$3"
MODEL="${4:-default}"
THRESHOLD="${5:-0.5}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

mkdir -p "$OUTPUT_DIR"

echo "开始关系提取..."
echo "输入目录: $INPUT_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "Prompt 文件: $PROMPT_FILE"
echo "模型: $MODEL"
echo "阈值: $THRESHOLD"
echo "------------------------"

python3 "$PROJECT_ROOT/main_cli.py" \
    --task relation_extraction \
    --relation_input_dir "$INPUT_DIR" \
    --relation_output_dir "$OUTPUT_DIR" \
    --relation_prompt "$PROMPT_FILE" \
    --relation_model "$MODEL" \
    --relation_threshold "$THRESHOLD"

echo "关系提取完成！结果保存在: $OUTPUT_DIR"