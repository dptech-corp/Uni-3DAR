# 8卡a100
[ -z "${MASTER_PORT}" ] && MASTER_PORT=10088
[ -z "${MASTER_IP}" ] && MASTER_IP=127.0.0.1
[ -z "${n_gpu}" ] && n_gpu=$(nvidia-smi -L | wc -l)
[ -z "${OMPI_COMM_WORLD_SIZE}" ] && OMPI_COMM_WORLD_SIZE=1
[ -z "${OMPI_COMM_WORLD_RANK}" ] && OMPI_COMM_WORLD_RANK=0
[ -z "${data_type}" ] && data_type=molecule
[ -z "${lr}" ] && lr=3e-4
[ -z "${min_lr}" ] && min_lr=1e-8
[ -z "${warmup_steps}" ] && warmup_steps=50000
[ -z "${total_steps}" ] && total_steps=500000
[ -z "${update_freq}" ] && update_freq=1
[ -z "${seed}" ] && seed=1
[ -z "${clip_norm}" ] && clip_norm=1
[ -z "${weight_decay}" ] && weight_decay=1e-4
[ -z "${merge_level}" ] && merge_level=8
[ -z "${lr_scheduler}" ] && lr_scheduler=cosine


[ -z "${layer}" ] && layer=12
[ -z "${batch_size}" ] && batch_size=16
[ -z "${emb_dim}" ] && emb_dim=768
[ -z "${head_num}" ] && head_num=12


data_path=$1
[ -z "${more_args}" ] && more_args=""


if [ "$lr_scheduler" = "cosine" ]; then
      more_args=$more_args" --lr-scheduler cosine --warmup-init-lr 1e-9 --min-lr $min_lr"
else
      more_args=$more_args" --lr-scheduler polynomial_decay --total-num-update $total_steps --end-learning-rate $min_lr"
fi

base_name=$2
[ -z "${base_dir}" ] && base_dir=./results
save_dir=$base_dir/$base_name
[ -z "${wandb_project}" ] && wandb_project=your_wandb_project

tmp_save_dir=/workspace/tmp_ckpt
mkdir -p $tmp_save_dir
mkdir -p $save_dir
cat $(pwd)/$0 > ${save_dir}/save_orders
printenv > ${save_dir}/environment_variables
log_save_dir=${save_dir}/log_${OMPI_COMM_WORLD_RANK}.txt
git rev-parse --abbrev-ref HEAD > ${save_dir}/git_info.txt
git log -1 >> ${save_dir}/git_info.txt
git log -1 
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
echo "n_gpu per node" $n_gpu
echo "OMPI_COMM_WORLD_SIZE" $OMPI_COMM_WORLD_SIZE
echo "OMPI_COMM_WORLD_RANK" $OMPI_COMM_WORLD_RANK
echo "MASTER_IP" $MASTER_IP
echo "MASTER_PORT" $MASTER_PORT


# comment out the following line if you want to use wandb
export WANDB_DISABLED=true
export WANDB_MODE=offline

torchrun --nproc_per_node=$n_gpu --nnodes=$OMPI_COMM_WORLD_SIZE  --node_rank=$OMPI_COMM_WORLD_RANK  --master_addr=$MASTER_IP --master_port=$MASTER_PORT \
      $(which unicore-train) $data_path --user-dir ./uni3dar --train-subset train --valid-subset valid \
      --num-workers 8 --ddp-backend=c10d \
      --task uni3dar --loss ar --arch uni3dar \
      --bf16 --tensorboard-logdir $save_dir/tsb \
      --wandb-project $wandb_project --wandb-name $base_name \
      --emb-dim $emb_dim --num-head $head_num  \
      --layer $layer \
      --log-interval 100 --log-format simple \
      --save-interval-updates 10000 --validate-interval-updates 10000 --keep-interval-updates 2 --no-epoch-checkpoints  \
      --save-dir $save_dir/ckpt --tmp-save-dir $tmp_save_dir \
      --batch-size $batch_size \
      --data-buffer-size 32 --fixed-validation-seed 11 --batch-size-valid $((batch_size * 2)) \
      --optimizer adam --adam-betas '(0.9, 0.99)' --adam-eps 1e-6 --clip-norm $clip_norm \
      --lr $lr --warmup-updates $warmup_steps --max-update $total_steps --update-freq $update_freq \
      --weight-decay $weight_decay \
      --seed $seed \
      --data-type $data_type --merge-level $merge_level \
      --ema-decay 0.999 --validate-with-ema \
      --grid-len 0.24  --xyz-resolution 0.01 --recycle 1  --tree-delete-ratio 0.4 --tree-delete-start-layer 1 --loss-ratio-tree 1.0 --loss-ratio-atom 1.0 --loss-ratio-xyz 1.0 --head-dropout 0.1 \
      $more_args \
      2>&1 | tee -a ${log_save_dir}

