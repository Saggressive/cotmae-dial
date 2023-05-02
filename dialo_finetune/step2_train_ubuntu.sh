export http_proxy=http://oversea-squid1.jp.txyun:11080 https_proxy=http://oversea-squid1.jp.txyun:11080 no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com
file=$1
dir=$2
name=${dir}_epoch5_lr5e-5_b64_correct_data
mkdir -p log/$name
nohup /share/miniconda3/envs/dialmodel/bin/python -m torch.distributed.launch --nproc_per_node 8 --nnodes=1 --master_port=23455 main.py \
--train_path data/ubuntu/train/data.json \
--val_path data/ubuntu/val/data.json \
--tensorboard_dir tflog/$name/$file \
--dataset ubuntu \
--num_eval_times 50 \
--model ../output/$dir/model/$file \
--save_path model/$name/$file \
--epochs 5 \
--eval_mode val \
--lr 5e-5 \
--batch_size 64 \
>log/$name/$file.log 2>&1 &
wait