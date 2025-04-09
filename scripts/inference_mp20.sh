[ -z "${MASTER_PORT}" ] && MASTER_PORT=10088
[ -z "${MASTER_IP}" ] && MASTER_IP=127.0.0.1
[ -z "${n_gpu}" ] && n_gpu=1
[ -z "${OMPI_COMM_WORLD_SIZE}" ] && OMPI_COMM_WORLD_SIZE=1
[ -z "${OMPI_COMM_WORLD_RANK}" ] && OMPI_COMM_WORLD_RANK=0

[ -z "${seed}" ] && seed=2
[ -z "${merge_level}" ] && merge_level=8

[ -z "${data_path}" ] && data_path=None
[ -z "${layer}" ] && layer=12
[ -z "${batch_size}" ] && batch_size=512
[ -z "${emb_dim}" ] && emb_dim=768
[ -z "${head_num}" ] && head_num=12

[ -z "${more_args}" ] && more_args=""


[ -z "${tree_temperature}" ] && tree_temperature=1.0
[ -z "${atom_temperature}" ] && atom_temperature=0.9
[ -z "${xyz_temperature}" ] && xyz_temperature=0.9
[ -z "${count_temperature}" ] && count_temperature=1.0
[ -z "${num_samples}" ] && num_samples=10000
[ -z "${rank_ratio}" ] && rank_ratio=0.32
[ -z "${rank_by}" ] && rank_by="atom+xyz"
[ -z "${data_type}" ] && data_type=crystal
[ -z "${save_path}" ] && save_path=$1_res_s${seed}_tt${tree_temperature}_at${atom_temperature}_xt${xyz_temperature}_ct${count_temperature}_ns${num_samples}_rr${rank_ratio}_rb${rank_by}.cif

echo "save_path" $save_path


export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1


torchrun --nproc_per_node=$n_gpu --nnodes=$OMPI_COMM_WORLD_SIZE  --node_rank=$OMPI_COMM_WORLD_RANK  --master_addr=$MASTER_IP --master_port=$MASTER_PORT \
      uni3dar/inference.py $data_path --user-dir ./uni3dar --train-subset train --valid-subset valid \
      --num-workers 8 --ddp-backend=c10d \
      --task uni3dar --loss ar --arch uni3dar_sampler \
      --bf16 \
      --emb-dim $emb_dim --num-head $head_num  \
      --layer $layer \
      --batch-size $batch_size\
      --data-buffer-size 32 --fixed-validation-seed 11 --batch-size-valid $((batch_size * 2)) \
      --seed $seed \
      --data-type $data_type --merge-level $merge_level  \
      --grid-len 0.24  --xyz-resolution 0.01 --recycle 2  \
      --tree-temperature $tree_temperature --atom-temperature $atom_temperature --xyz-temperature $xyz_temperature --count-temperature $count_temperature \
      --num-samples $num_samples --rank-ratio $rank_ratio --rank-by $rank_by \
      --save-path $save_path --gzip \
      --atom-type-key atom_type --atom-pos-key atom_pos --lattice-matrix-key lattice_matrix --allow-atoms all  --head-dropout 0.1 \
      --max-num-atom 128 \
      --finetune-from-model $1 \
      $more_args


python evaluation_scripts/crystal/crystal_metrics.py $save_path $save_path.json mp20 $2
