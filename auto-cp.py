import os
import subprocess
import time

# 配置相关变量
REPO_PATH = "/root/source/oceanbase"  # oceanbase 仓库路径
BUILD_COMMAND = "bash build.sh release --init --make"  # 编译命令
STOP_COMMAND = "./tools/deploy/obd.sh stop -n obcluster"  # 集群停止命令
SOURCE_FILE = "build_release/src/observer/observer"  # CP源文件路径
DESTINATION_FILE = "/data/obcluster/bin/observer"  #CP 目标文件路径
START_COMMAND = "./tools/deploy/obd.sh start -n obcluster"  # START 命令
ANN_REPO_PATH = "/root/source/ann-benchmarks"  # ANN 仓库路径
ANN_BRANCH_NAME = "oceanbase_public"  # ANN目标分支名
# ANN-Benchmarks 配置
ANN_BENCHMARKS_PATH = "/root/source/ann-benchmarks"  # ann-benchmarks 路径
DATASET = "sift-128-euclidean"  # 数据集
ALGORITHM = "oceanbase"  # 使用的算法
RUNS = 1  # 运行次数

# 导入数据并构建索引命令
IMPORT_COMMAND = f"python run.py --algorithm {ALGORITHM} --local --force --dataset {DATASET} --runs {RUNS}"

# 执行三次 skip_fit 测试命令
SKIP_FIT_COMMAND = f"python run.py --algorithm {ALGORITHM} --local --force --dataset {DATASET} --runs {RUNS} --skip_fit"

# 计算召回率及 QPS
PLOT_COMMAND = f"python plot.py --dataset {DATASET} --recompute"

def run_command(command, cwd=None):
    """
    执行系统命令并打印输出
    """
    try:
        print(f"执行命令: {command}")
        process = subprocess.Popen(
            command, shell=True, cwd=cwd,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
        )

        # 实时读取标准输出
        for line in iter(process.stdout.readline, ''):
            print(f"标准输出: {line.strip()}")  # 实时打印标准输出

        # 等待子进程完成并获取结果
        stdout, stderr = process.communicate()
        print("标准输出:")
        print(stdout)
        print("错误输出:")
        print(stderr)

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command, stdout, stderr)

    except subprocess.CalledProcessError as e:
        print(f"命令失败: {e.stderr}")
        raise
    except Exception as e:
        print(f"发生未知错误: {str(e)}")
        raise

def copy_file(source, destination):
    """
    复制文件
    :param source: 源文件路径
    :param destination: 目标文件路径
    """
    try:
        print(f"复制文件: 从 {source} 到 {destination}")
        os.makedirs(os.path.dirname(destination), exist_ok=True)  # 确保目标目录存在
        subprocess.run(["cp", source, destination], check=True)
        print("文件复制完成。")
    except Exception as e:
        print(f"文件复制失败: {str(e)}")
        raise

def main():
    print("=== 开始操作 ===")
    try:
        # 1. 获取分支名字
        branch_name = input("请输入分支名字: ").strip()
        if not branch_name:
            raise ValueError("分支名字不能为空！")

        # 拉取最新代码并切换分支
        print("=== 拉取最新代码并切换分支 ===")
        run_command("git fetch", cwd=REPO_PATH)
        run_command(f"git checkout {branch_name}", cwd=REPO_PATH)

        # 2. 执行编译命令
        print("=== 执行编译命令 ===")
        run_command(BUILD_COMMAND, cwd=REPO_PATH)
        time.sleep(30)  # 等待 30 秒

        # 3. 停止服务
        print("=== 停止服务 ===")
        run_command(STOP_COMMAND, cwd=REPO_PATH)
        time.sleep(10)

        # 4. 复制文件
        print("=== 复制文件 ===")
        copy_file(SOURCE_FILE, DESTINATION_FILE)
        time.sleep(10)

         # 4. 执行 start 命令
        print("=== 执行 start 命令 ===")
        run_command(START_COMMAND, cwd=REPO_PATH)
        time.sleep(30)  # 等待 30 秒

        # # 7. 拉取最新ANN代码并切换分支
        run_command("git fetch", cwd=ANN_REPO_PATH)
        run_command(f"git checkout {ANN_BRANCH_NAME}", cwd=ANN_REPO_PATH)
        time.sleep(5)

        # 8. 导入数据并构建索引
        print("=== 导入数据并构建索引 ===")
        run_command(IMPORT_COMMAND, cwd=ANN_BENCHMARKS_PATH)
        time.sleep(10)  # 等待 10 秒

        # 9. 执行三次 skip_fit 测试命令
        print("=== 执行三次 skip_fit 测试命令 ===")
        for _ in range(3):
            run_command(SKIP_FIT_COMMAND, cwd=ANN_BENCHMARKS_PATH)

        # 10. 计算召回率及 QPS
        time.sleep(10)  # 等待 10 秒
        print("=== 计算召回率及 QPS ===")
        run_command(PLOT_COMMAND, cwd=ANN_BENCHMARKS_PATH)
        print("=== 所有操作完成 ===")
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    main()