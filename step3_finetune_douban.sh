en=$1
de=$2
max_steps=150000
lr=$3
dir=douban
nhead=1
# file=step${max_steps}_en${en}_de${de}_lr${lr}_nhead${nhead}
file=step150000_en0.15_disable_decoder_lr1e-4
cd dialo_finetune
bash step2_train_douban.sh $file $dir