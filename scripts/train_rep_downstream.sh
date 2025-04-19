[ -z "${MASTER_PORT}" ] && MASTER_PORT=10088
[ -z "${MASTER_IP}" ] && MASTER_IP=127.0.0.1
[ -z "${n_gpu}" ] && n_gpu=$(nvidia-smi -L | wc -l)
[ -z "${OMPI_COMM_WORLD_SIZE}" ] && OMPI_COMM_WORLD_SIZE=1
[ -z "${OMPI_COMM_WORLD_RANK}" ] && OMPI_COMM_WORLD_RANK=0
[ -z "${min_lr}" ] && min_lr=1e-9

[ -z "${clip_norm}" ] && clip_norm=1
[ -z "${weight_decay}" ] && weight_decay=1e-4
[ -z "${lr_scheduler}" ] && lr_scheduler=cosine

[ -z "${lr}" ] && lr=1e-4
[ -z "${update_freq}" ] && update_freq=1
[ -z "${seed}" ] && seed=1
[ -z "${epoch}" ] && epoch=40
[ -z "${warmup}" ] && warmup=0.06
[ -z "${pooler_dropout}" ] && pooler_dropout=0.0
[ -z "${batch_size}" ] && batch_size=8
[ -z "${adam_betas}" ] && adam_betas="'(0.9, 0.99)'"
[ -z "${valid_subset}" ] && valid_subset=valid,test



data_path=$1
[ -z "${more_args}" ] && more_args=""

if [ "$lr_scheduler" = "cosine" ]; then
    more_args=$more_args" --lr-scheduler cosine --warmup-init-lr 1e-9 --min-lr $min_lr"
else
    more_args=$more_args" --lr-scheduler polynomial_decay --end-learning-rate $min_lr"
fi
# set -e

task_name=$3
weight_path=$4
lr_list=$5
batch_size_list=$6
epoch_list=$7
pooler_dropout_list=$8
warmup_list=$9
seed_list=${10}

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
echo "n_gpu per node" $n_gpu
echo "OMPI_COMM_WORLD_SIZE" $OMPI_COMM_WORLD_SIZE
echo "OMPI_COMM_WORLD_RANK" $OMPI_COMM_WORLD_RANK
echo "MASTER_IP" $MASTER_IP
echo "MASTER_PORT" $MASTER_PORT
echo $task_name

[ -z "${wandb_project}" ] && wandb_project=your_wandb_project


if [ "${task_name}" == "homo" ] || [ "${task_name}" == "lumo" ] || [ "${task_name}" == "gap" ] || [ "${task_name}" == "E1_CC2" ] || [ "${task_name}" == "E2_CC2" ] || [ "${task_name}" == "f1_CC2" ] || [ "${task_name}" == "f2_CC2" ] || [ "${task_name}" == "Dipmom_Debye" ] || [ "${task_name}" == "aIP_eV" ] || [ "${task_name}" == "D3_disp_corr_eV" ] ; then
	# 单卡a100
    metric="valid_mol_target_reg_agg_mae"
    batch_size_t=64
    n_gpu=1
    more_args=$more_args" --merge-level 8 --data-type molecule --grid-len 0.24 --recycle 1 --mol-target-key target --mol-num-classes 1 --mol-target-normalize --loss-ratio-mol-target 50"
    if [ "${task_name}" == "homo" ]; then
        more_args=$more_args" --mol-target-idx 2"
    elif [ "${task_name}" == "lumo" ]; then
        more_args=$more_args" --mol-target-idx 3"
    elif [ "${task_name}" == "gap" ]; then
        more_args=$more_args" --mol-target-idx 4"
    elif [ "${task_name}" == "E1_CC2" ]; then
        more_args=$more_args" --mol-target-idx 0"
    elif [ "${task_name}" == "E2_CC2" ]; then
        more_args=$more_args" --mol-target-idx 1"
    elif [ "${task_name}" == "f1_CC2" ]; then
        more_args=$more_args" --mol-target-idx 2"
    elif [ "${task_name}" == "f2_CC2" ]; then
        more_args=$more_args" --mol-target-idx 3"
    elif [ "${task_name}" == "Dipmom_Debye" ]; then
        more_args=$more_args" --mol-target-idx 3"
    elif [ "${task_name}" == "aIP_eV" ]; then
        more_args=$more_args" --mol-target-idx 9"
    elif [ "${task_name}" == "D3_disp_corr_eV" ]; then
        more_args=$more_args" --mol-target-idx 10"
    fi
    
elif  [ "$task_name" == "binding" ]; then
    # 8卡a100
    metric="valid_atom_target_cls_agg_auc"
    batch_size_t=4
    n_gpu=8
    more_args=${more_args}" --merge-level 10 --data-type protein --grid-len 0.48 --recycle 2 --maximize-best-checkpoint-metric --atom-target-key pocket_label --atom-num-classes 2 --loss-ratio-atom-target 5 --gzip --all-gather-list-size 1638400"
    valid_subset=ASTEX85,valid,COACH420,CHEN251,B277,DT198
fi

IFS=' ' read -r -a lr_list <<< "$lr_list"
IFS=' ' read -r -a batch_size_list <<< "$batch_size_list"
IFS=' ' read -r -a epoch_list <<< "$epoch_list"
IFS=' ' read -r -a pooler_dropout_list <<< "$pooler_dropout_list"
IFS=' ' read -r -a warmup_list <<< "$warmup_list"
IFS=' ' read -r -a seed_list <<< "$seed_list"
target_string="Program finished ..."

echo ${lr_list[@]}
echo ${batch_size_list[@]}
echo ${epoch_list[@]}
echo ${pooler_dropout_list[@]}
echo ${warmup_list[@]}
echo ${seed_list[@]}

# comment out the following line if you want to use wandb
export WANDB_DISABLED=true
export WANDB_MODE=offline

for lr in ${lr_list[@]}
    do for batch_size in ${batch_size_list[@]}
        do for epoch in ${epoch_list[@]}
            do for pooler_dropout in ${pooler_dropout_list[@]}
                do for warmup in ${warmup_list[@]}
                    do for seed in ${seed_list[@]}
                        do
                            if [ "$batch_size_t" -lt "$batch_size" ]; then
                                batch_size_t=$batch_size_t
                            else
                                batch_size_t=$batch_size
                            fi
                            update_freq=$(expr $batch_size / $batch_size_t  / $n_gpu )
                            echo "batch_size" $batch_size
                            echo "batch_size_per_gpu" $batch_size_t
                            echo "update_freq" $update_freq
                            total_bs=$((batch_size_t * n_gpu * update_freq))
                            echo "task_name" $task_name
                            echo "lr" $lr
                            echo "bsz" $total_bs
                            echo "epoch" $epoch
                            echo "dropout" $pooler_dropout
                            echo "warmup" $warmup
                            echo "seed" $seed
                            echo "adam_betas" $adam_betas
                            base_name=${task_name}_lr${lr}_bsz${total_bs}_epoch${epoch}_dropout${pooler_dropout}_warmup${warmup}_seed${seed}
                            [ -z "${base_dir}" ] && base_dir=./results/$2
                            save_dir=$base_dir/$base_name
                            echo "save_dir" $save_dir
                            log_save_dir=${save_dir}/log.txt
                            last_line='empty'
                            second_last_line='empty'
                            if [ -f "$log_save_dir" ]; then
                                echo "File exists: $log_save_dir"
                                if grep -q "INFO | unicore_cli.train | done training" $log_save_dir; then
                                    found=1
                                    echo "Experiment finished!"
                                else
                                    found=0
                                fi
                            else
                                echo "File not exists: $log_save_dir"
                                found=0                                                              
                            fi
                            FILE_PATH=${save_dir}/ckpt/checkpoint_last.pt
                            if [ -f "$FILE_PATH" ]; then
                                echo "File exists..."
                                more_args_new=${more_args}
                            else
                                echo "File not exists $FILE_PATH"
                                more_args_new=${more_args}"  --finetune-from-model $weight_path --load-from-ema"
                            fi
                            echo "more args ...${more_args_new}"


                            train_args="--warmup-ratio $warmup --max-epoch $epoch --validate-interval 1 --keep-last-epochs 1 --save-interval 10 --log-interval 10"
                            
                            if [ "$found" -eq 0 ]; then

                                mkdir -p $save_dir
                                cat $(pwd)/$0 > ${save_dir}/save_orders
                                printenv > ${save_dir}/environment_variables
                                git rev-parse --abbrev-ref HEAD > ${save_dir}/git_info.txt
                                git log -1 >> ${save_dir}/git_info.txt

                                eval torchrun --nproc_per_node=$n_gpu --nnodes=$OMPI_COMM_WORLD_SIZE  --node_rank=$OMPI_COMM_WORLD_RANK  --master_addr=$MASTER_IP --master_port=$MASTER_PORT \
                                    $(which unicore-train) $data_path --user-dir ./uni3dar --train-subset train --valid-subset $valid_subset \
                                    --num-workers 8 --ddp-backend=c10d \
                                    --task uni3dar --loss ar --arch uni3dar \
                                    --bf16 --tensorboard-logdir $save_dir/tsb \
                                    --wandb-project $wandb_project --wandb-name $base_name \
                                    --log-format simple \
                                    --save-dir $save_dir/ckpt \
                                    --batch-size $batch_size_t \
                                    --data-buffer-size 32 --fixed-validation-seed 11 --batch-size-valid $(expr $batch_size_t \* 2) \
                                    --optimizer adam --adam-betas "$adam_betas" --adam-eps 1e-6 --clip-norm $clip_norm \
                                    --lr $lr \
                                    --weight-decay $weight_decay \
                                    --seed $seed  \
                                    --checkpoint-activation-threshold 100000  --expand-valid-dataset \
                                    --update-freq $update_freq --best-checkpoint-metric $metric --pooler-dropout $pooler_dropout \
                                    --task-name $task_name --finetune \
                                    --layer 12   --H-prob 1.0  --repeat-count 5  --emb-dim 768 --num-head 12  \
                                    --xyz-resolution 0.01    --tree-delete-ratio 0.4  --tree-delete-start-layer 1 \
                                    --loss-ratio-tree 1.0 --loss-ratio-atom 1.0   --loss-ratio-xyz 1.0 --head-dropout 0.1 --no-save  \
                                    $train_args \
                                    $more_args_new \
                                    2>&1 | tee -a ${log_save_dir}
                            fi
                        done
                    done
                done
            done
        done
    done