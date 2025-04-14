#!/usr/bin/env bash

# 检查是否提供了模型名称
if [ -z "$1" ]; then
    echo "请提供模型名称"
    echo "用法: $0 <model_name>"
    echo "例如: $0 gpt2"
    exit 1
fi

# 设置模型名称
MODEL_NAME="$1"

# 创建别名
alias hfd="$PWD/hfd.sh"

# 格式化缓存目录名称
# 将斜杠替换为--
CACHE_DIR_NAME=$(echo "$MODEL_NAME" | sed 's/\//-/g')
CACHE_PATH="$HOME/.cache/huggingface/hub/models--${CACHE_DIR_NAME}"

# 创建缓存目录
mkdir -p "$CACHE_PATH"

# 调用hfd命令
"$PWD/hfd.sh" "$MODEL_NAME" --exclude "onnx/*" --local-dir "$CACHE_PATH"