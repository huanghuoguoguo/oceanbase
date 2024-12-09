#!/bin/bash

cd /root/source/ann-benchmarks
python run.py --algorithm oceanbase --local --force --dataset sift-128-euclidean --runs 1
python plot.py --dataset sift-128-euclidean --recompute

python run.py --algorithm oceanbase --local --force --dataset sift-128-euclidean --runs 1 --skip_fit
python plot.py --dataset sift-128-euclidean --recompute

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