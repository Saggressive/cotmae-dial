max_steps=150000
lr=3e-4
dir=ubuntu
nhead=1
cd dialo_finetune

file=step${max_steps}_en0.3_de0.75_lr${lr}_nhead${nhead}_mul
bash step2_train_ubuntu.sh $file $dir


file=step${max_steps}_en0.45_de0.75_lr${lr}_nhead${nhead}
bash step2_train_ubuntu.sh $file $dir

