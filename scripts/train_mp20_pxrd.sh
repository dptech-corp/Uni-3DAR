[ -z "${MASTER_PORT}" ] && MASTER_PORT=10088
[ -z "${MASTER_IP}" ] && MASTER_IP=127.0.0.1
[ -z "${n_gpu}" ] && n_gpu=$(nvidia-smi -L | wc -l)
[ -z "${OMPI_COMM_WORLD_SIZE}" ] && OMPI_COMM_WORLD_SIZE=1
[ -z "${OMPI_COMM_WORLD_RANK}" ] && OMPI_COMM_WORLD_RANK=0
[ -z "${data_type}" ] && data_type=crystal
[ -z "${lr}" ] && lr=3e-4
[ -z "${min_lr}" ] && min_lr=1e-9
[ -z "${warmup_steps}" ] && warmup_steps=40000
[ -z "${total_steps}" ] && total_steps=400000
[ -z "${update_freq}" ] && update_freq=1
[ -z "${seed}" ] && seed=1
[ -z "${clip_norm}" ] && clip_norm=1
[ -z "${weight_decay}" ] && weight_decay=1e-4
[ -z "${merge_level}" ] && merge_level=8


[ -z "${layer}" ] && layer=24
[ -z "${batch_size}" ] && batch_size=16
[ -z "${emb_dim}" ] && emb_dim=1024
[ -z "${head_num}" ] && head_num=16


data_path=$1
[ -z "${more_args}" ] && more_args=""


echo "more_args" $more_args

[ -z "${base_dir}" ] && base_dir=./results
base_name=$2
save_dir=$base_dir/$base_name
[ -z "${wandb_project}" ] && wandb_project=your_wandb_project

tmp_save_dir=./tmp_ckpt
mkdir -p $tmp_save_dir
mkdir -p $save_dir
cat $(pwd)/$0 > ${save_dir}/save_orders
printenv > ${save_dir}/environment_variables
log_save_dir=${save_dir}/log_${OMPI_COMM_WORLD_RANK}.txt
git log -1 >> ${save_dir}/git_info.txt
git log -1 
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
echo "n_gpu per node" $n_gpu
echo "OMPI_COMM_WORLD_SIZE" $OMPI_COMM_WORLD_SIZE
echo "OMPI_COMM_WORLD_RANK" $OMPI_COMM_WORLD_RANK
echo "MASTER_IP" $MASTER_IP
echo "MASTER_PORT" $MASTER_PORT

set -o pipefail

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
      --lr $lr --update-freq $update_freq \
      --weight-decay $weight_decay \
      --seed $seed \
      --data-type $data_type --merge-level $merge_level  \
      --warmup-updates $warmup_steps --max-update $total_steps \
      --ema-decay 0.999 --validate-with-ema \
      --lr-scheduler cosine --warmup-init-lr 1e-9 --min-lr $min_lr \
      --grid-len 0.24  --gzip --H-prob 1.0 --xyz-resolution 0.01 --recycle 1  \
      --loss-ratio-tree 0.5,1.0 --loss-ratio-atom 0.25,0.5 --loss-ratio-xyz 0.05,0.1 --loss-ratio-count 0.5,1.0 \
      --atom-type-key atom_type --atom-pos-key atom_pos --lattice-matrix-key lattice_matrix --allow-atoms all  --head-dropout 0.1 \
      --tree-delete-start-layer 1 --tree-delete-ratio 0.1  --crystal-pxrd 4 --crystal-pxrd-step 0.1 --crystal-pxrd-noise 0.1 \
      --crystal-component 1 --crystal-component-sqrt --crystal-component-noise 0.1 --crystal-pxrd-threshold 5 --max-num-atom 128 --random-rotation-prob 0.5 --crystal-random-shift-prob 0.5 \
      $more_args \
      2>&1 | tee -a ${log_save_dir}

exit $?