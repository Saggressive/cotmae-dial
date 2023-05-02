#!/bin/bash
mode=$1
if [ $mode == "ubuntu" ];then
    echo "ubuntu"
    python make_data.py \
    --tokenizer bert-base-uncased \
    --save_to data/ubuntu \
    --data_path data/ubuntu
elif [ $mode == "douban" ];then
    echo "douban"
    python make_data.py \
    --tokenizer bert-base-chinese \
    --save_to data/douban \
    --data_path data/douban
elif [ $mode == "ecommerce" ];then
    echo "ecommerce"
    python make_data.py \
    --tokenizer bert-base-chinese \
    --save_to data/ecommerce_balance \
    --data_path data/ecommerce_balance
elif [ $mode == "rrs" ];then
    echo "rrs"
    python make_data.py \
    --tokenizer bert-base-chinese \
    --save_to data/rrs \
    --data_path data/rrs
else
    echo "error"
fi