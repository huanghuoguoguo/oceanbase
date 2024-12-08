#!/bin/bash
git pull
# Stop the obcluster
./tools/deploy/obd.sh stop -n obcluster

rm -rf /data/obcluster/log/*

# Build the release
bash build.sh release --init --make --silent
if [ $? -ne 0 ]; then
  echo "Build failed. Exiting."
  exit 1
fi

# Force copy of the observer binary
cp -f build_release/src/observer/observer /data/obcluster/bin/observer
if [ $? -ne 0 ]; then
  echo "Failed to copy observer. Exiting."
  exit 1
fi

# Start the obcluster
./tools/deploy/obd.sh start -n obcluster
if [ $? -ne 0 ]; then
  echo "Failed to start obcluster. Exiting."
  exit 1
fi

# Wait for 30 seconds
echo "Waiting for 30 seconds..."
sleep 30


# Run the benchmark
cd /root/source/ann-benchmarks
python run.py --algorithm oceanbase --local --force --dataset sift-128-euclidean --runs 1
if [ $? -ne 0 ]; then
  echo "Benchmark run failed. Exiting."
  exit 1
fi

# Plot the results
python plot.py --dataset sift-128-euclidean --recompute
if [ $? -ne 0 ]; then
  echo "Plotting failed. Exiting."
  exit 1
fi

python run.py --algorithm oceanbase --local --force --dataset sift-128-euclidean --runs 1 --skip_fit
python plot.py --dataset sift-128-euclidean --recompute

python run.py --algorithm oceanbase --local --force --dataset sift-128-euclidean --runs 1 --skip_fit
python plot.py --dataset sift-128-euclidean --recompute

cd /root/source/oceanbase
./tools/deploy/obd.sh restart -n obcluster
cd /root/source/ann-benchmarks

python run.py --algorithm oceanbase --local --force --dataset sift-128-euclidean --runs 1 --skip_fit
python plot.py --dataset sift-128-euclidean --recompute

cd /root/source/ann-benchmarks/ann_benchmarks/algorithms/oceanbase
python hybrid_ann.py


# Change to oceanbase source directory
cd /root/source/oceanbase


sleep 10

# Stop the obcluster
./tools/deploy/obd.sh stop -n obcluster
if [ $? -ne 0 ]; then
  echo "Failed to stop obcluster. Exiting."
  exit 1
fi

echo "Deployment, testing, and stopping of obcluster completed successfully."
