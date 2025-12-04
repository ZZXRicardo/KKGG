#!/bin/bash

# ===========================================
# 实体提取脚本（严格对齐 Python 参数）
# 功能：日志 + 断点续跑 + 错误存储
#
# 用法:
#   ./extract_entities.sh <input_dir> <output_dir> <prompt_file> [options...]
#
# 必填（3个）:
#   input_dir      : 输入 JSON 目录
#   output_dir     : 输出目录
#   prompt_file    : Prompt 文件路径
#
# 可选（按顺序）:
#   [model]         → 默认: qianwen
#   [threshold]     → 默认: 0.5
#   [start_index]   → 默认: 0
#   [end_index]     → 默认: -1
#   [resume]        → 若为 "resume" 则启用 --resume
#   [log_dir]       → 默认: ./logs
#
# 示例:
#   ./extract_entities.sh data/ out/ prompts/entity.txt qianwen 0.5 0 -1 resume ./mylogs
# ===========================================

set -e

if [ "$#" -lt 3 ]; then
    echo "错误：至少需要 3 个参数"
    echo "用法: $0 <input_dir> <output_dir> <prompt_file> [model] [threshold] [start_index] [end_index] [resume] [log_dir]"
    echo ""
    echo "参数说明:"
    echo "  input_dir   : 输入 JSON 目录（必须存在）"
    echo "  output_dir  : 输出目录（将自动创建）"
    echo "  prompt_file : Prompt 模板文件（必须存在）"
    echo "  model       : 模型名（默认: qianwen）"
    echo "  threshold   : 置信度阈值（默认: 0.5）"
    echo "  start_index : 起始文件索引（默认: 0）"
    echo "  end_index   : 结束索引（默认: -1）"
    echo "  resume      : 传 'resume' 启用断点续跑"
    echo "  log_dir     : 日志目录（默认: ./logs）"
    exit 1
fi

# === 解析参数 ===
INPUT_DIR="$(realpath "$1")"
OUTPUT_DIR="$(realpath "$2")"
PROMPT_FILE="$(realpath "$3")"
MODEL="${4:-qianwen}"
THRESHOLD="${5:-0.5}"
START_INDEX="${6:-0}"
END_INDEX="${7:--1}"
RESUME_ARG="${8:-}"
LOG_DIR="${9:-./logs}"

# === 验证输入 ===
if [ ! -d "$INPUT_DIR" ]; then
    echo "❌ 错误：输入目录不存在: $INPUT_DIR" >&2
    exit 1
fi

if [ ! -f "$PROMPT_FILE" ]; then
    echo "❌ 错误：Prompt 文件不存在: $PROMPT_FILE" >&2
    exit 1
fi

# === 处理 resume 标志 ===
RESUME_FLAG=""
if [ "$RESUME_ARG" = "resume" ]; then
    RESUME_FLAG="--resume"
fi

# === 路径处理 ===
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# === 打印配置 ===
echo "启动实体提取任务"
echo "------------------------"
echo "输入目录     : $INPUT_DIR"
echo "输出目录     : $OUTPUT_DIR"
echo "Prompt 文件  : $PROMPT_FILE"
echo "模型         : $MODEL"
echo "阈值         : $THRESHOLD"
echo "起始索引     : $START_INDEX"
echo "结束索引     : $END_INDEX"
echo "断点续跑     : $([ -n "$RESUME_FLAG" ] && echo "是" || echo "否")"
echo "日志目录     : $(realpath "$LOG_DIR")"
echo "------------------------"

# === 调用主程序（严格使用已定义参数）===
python "$PROJECT_ROOT/main_cli.py" \
    --task entity_extraction \
    --entity_input_dir "$INPUT_DIR" \
    --entity_output_dir "$OUTPUT_DIR" \
    --entity_prompt "$PROMPT_FILE" \
    --entity_model "$MODEL" \
    --entity_threshold "$THRESHOLD" \
    --start_index "$START_INDEX" \
    --end_index "$END_INDEX" \
    --log_dir "$LOG_DIR" \
    $RESUME_FLAG

echo "实体提取完成！"
echo "输出: $OUTPUT_DIR"
echo "日志: $(realpath "$LOG_DIR")"
# 错误目录由 Python 自动设为 <output_dir>/errors，无需额外指定
