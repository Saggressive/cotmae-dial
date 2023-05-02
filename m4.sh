max_steps=150000
lr=4e-4
dir=ubuntu
cd dialo_finetune

file=step${max_steps}_en0.3_de0.45_lr${lr}_nhead2
bash step2_train_ubuntu.sh $file $dir

file=step${max_steps}_en0.3_de0.45_lr${lr}_nhead4
bash step2_train_ubuntu.sh $file $dir

