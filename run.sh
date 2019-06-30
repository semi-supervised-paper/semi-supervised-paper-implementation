#!/usr/bin/env bash
#rlaunch --cpu=2 --gpu=1 --memory=10000 -- python3 -m semi_supervised.experiments.tempens.cifar10_test &
#rlaunch --cpu=2 --gpu=1 --memory=10000 -- python3 -m semi_supervised.experiments.pi.cifar10_test &
#rlaunch --cpu=2 --gpu=1 --memory=10000 -- python3 -m semi_supervised.experiments.mean_teacher.cifar10_test &
rlaunch --cpu=2 --gpu=1 --memory=10000 -- python3 -m semi_supervised.experiments.vat.cifar10_test
wait
echo "Finished"
