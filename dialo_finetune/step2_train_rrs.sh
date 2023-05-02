export http_proxy=http://oversea-squid1.jp.txyun:11080 https_proxy=http://oversea-squid1.jp.txyun:11080 no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com
file=$1
dir=$2
mkdir -p log/${dir}_rrs
nohup /share/miniconda3/envs/dialmodel/bin/python -m torch.distributed.launch --nproc_per_node 8 --nnodes=1 --master_port=23455 main.py \
--train_path data/rrs/train/data.json \
--val_path data/rrs/val/data.json \
--tensorboard_dir tflog/${dir}_rrs/$file \
--dataset rrs \
--num_eval_times 100 \
--model ../output/$dir/model/$file \
--save_path model/${dir}_rrs/$file \
--epochs 5 \
>log/${dir}_rrs/$file.log 2>&1 &
wait