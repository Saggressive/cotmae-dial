export http_proxy=http://oversea-squid1.jp.txyun:11080 https_proxy=http://oversea-squid1.jp.txyun:11080 no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com
# Enable RDMA
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
# export NCCL_MIN_NCHANNELS=16
export NCCL_SOCKET_IFNAME=eth
export NCCL_IB_HCA=mlx5
en=$1
de=$2
max_steps=$3
lr=$4
nhead=$5
node_rank=$6
file=step${max_steps}_en${en}_de${de}_lr${lr}_nhead${nhead}_mul
model_dir=output/ubuntu/model/$file
tflog_dir=output/ubuntu/tflog_dir
log_dir=output/ubuntu/log
mkdir -p $model_dir
mkdir -p $tflog_dir
mkdir -p $log_dir
nohup /share/miniconda3/envs/dialmodel/bin/python -m torch.distributed.launch --nnodes=2 --master_addr=10.80.208.23 --nproc_per_node 8 --master_port 3242 --node_rank=${node_rank} run_pretraining.py \
--model_name_or_path bert-base-uncased \
--output_dir $model_dir \
--do_train \
--logging_steps 200 \
--save_steps 20000 \
--save_total_limit 4 \
--fp16 \
--warmup_ratio 0.1 \
--logging_dir $tflog_dir/$file \
--per_device_train_batch_size 64 \
--gradient_accumulation_steps 1 \
--learning_rate $lr \
--max_steps $max_steps \
--overwrite_output_dir \
--dataloader_drop_last \
--dataloader_num_workers 16 \
--context_seq_length 256 \
--response_seq_length 64 \
--train_path data/ubuntu/train/data.json \
--weight_decay 0.01 \
--encoder_mask_ratio $en \
--decoder_mask_ratio $de \
--use_decoder_head \
--enable_head_mlm \
--n_head_layers $nhead \
>$log_dir/${file}_${node_rank}.log 2>&1 &