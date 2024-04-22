#!/bin/bash

python -m benchmark_models.injector.tf_weight_injection -n DenseNet121 -d CIFAR100 --fault-list fault_lists/CIFAR100/DenseNet121_CIFAR100_fault_list.csv --sort-tf-layers -b 512