#!/bin/bash

# 检查是否提供了参数
if [ $# -eq 0 ]; then
    echo "请提供执行部分的参数（1 或 2）"
    exit 1
fi

PART=$1

# 执行第1部分
if [ $PART -eq 1 ]; then
    echo "请输入更新说明："
    read commit_message
    commit_message=${commit_message:-"update"}

    git add .
    git commit -m "$commit_message"
    git push origin main
fi

# 执行第2部分
if [ $PART -eq 2 ]; then
    echo "正在从远程仓库拉取更新..."
    git pull origin main

    if [ $? -eq 0 ]; then
        echo "更新成功！"
    else
        echo "更新失败，请检查错误信息"
    fi
fi