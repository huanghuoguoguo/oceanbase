#!/bin/bash

# 设置工作目录为 /root/source/oceanbase
cd /root/source/oceanbase || exit 1

# 执行 obd.sh start -n obcluster 来启动 obcluster
echo "启动 obcluster..."
./tools/deploy/obd.sh start -n obcluster

# 检查 obd.sh start 命令是否执行成功
if [ $? -ne 0 ]; then
    echo "启动 obcluster 失败，脚本终止！"
    exit 1
fi

# 如果成功启动 obcluster，继续执行后续步骤

# 检查当前环境是否为 ann，若不是则激活 ann 环境
if [[ "$CONDA_DEFAULT_ENV" != "ann" ]]; then
    echo "当前环境不是 ann，激活 ann 环境..."
    conda activate ann
fi

# 设置工作目录为 /root/source/ann-benchmarks
cd /root/source/ann-benchmarks || exit 1

# 执行第一个 Python 脚本
echo "执行第一个 Python 脚本..."
python run.py --algorithm oceanbase --local --force --dataset sift-128-euclidean --runs 1

# 执行第二个 Python 脚本
echo "执行第二个 Python 脚本..."
python run.py --algorithm oceanbase --local --force --dataset sift-128-euclidean --runs 1 --skip_fit

# 执行 plot.py 脚本
echo "执行 plot.py 脚本..."
python plot.py --dataset sift-128-euclidean --recompute

# 输出日志到文件 log
echo "所有操作完成，结果已输出到 log 文件"
{
    echo "Running scripts in ann-benchmarks..."
    python run.py --algorithm oceanbase --local --force --dataset sift-128-euclidean --runs 1
    python run.py --algorithm oceanbase --local --force --dataset sift-128-euclidean --runs 1 --skip_fit
    python plot.py --dataset sift-128-euclidean --recompute
} > log 2>&1

# 设置工作目录为 /root/source/oceanbase
cd /root/source/oceanbase || exit 1

# 执行 obd.sh stop -n obcluster 停止 obcluster
echo "停止 obcluster..."
./tools/deploy/obd.sh stop -n obcluster

echo "脚本执行完毕！"
