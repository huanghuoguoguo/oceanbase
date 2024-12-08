#!/bin/bash

# 检查是否提供了搜索字符串
if [ -z "$1" ]; then
  echo "Usage: $0 <search_string>"
  exit 1
fi

# 获取用户输入的字符串
search_string="$1"

# 设置日志文件目录
log_dir="/data/obcluster/log"

# 检查目录是否存在
if [ ! -d "$log_dir" ]; then
  echo "Error: Directory $log_dir does not exist."
  exit 1
fi

# 使用 grep 搜索指定的字符串，遍历日志目录下的所有文件
echo "Searching for '$search_string' in logs at $log_dir..."
grep -r "$search_string" "$log_dir"/*

# 检查 grep 命令是否成功执行
if [ $? -ne 0 ]; then
  echo "No matches found for '$search_string'."
else
  echo "Search complete."
fi
