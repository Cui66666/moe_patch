# 8 * 57GiB, 2.95s/it
# export MEGATRON_LM_PATH=<your_local_megatron_lm_path>

# (Optional) Enable MoE load monitor
export PYTHONPATH="src/patch/swift:$PYTHONPATH"
export MOE_PATCH_DIR="logs/moe_monitor"

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
megatron sft \
    --model Qwen/Qwen3-30B-A3B-Base \
    --model_type qwen3_moe \
    --load_safetensors true \
    --save_safetensors true \
    --dataset data/OpenR1-Math-220k \
    --split_dataset_ratio 0.001 \
    --load_from_cache_file true \
    --pipeline_model_parallel_size 2 \
    --decoder_last_pipeline_num_layers 23 \
    --tensor_model_parallel_size 2 \
    --expert_model_parallel_size 2 \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-2 \
    --micro_batch_size 1 \
    --global_batch_size 16 \
    --packing true \
    --moe_router_dtype fp32 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --max_epochs 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --save megatron_output/Qwen3-MoE-A3B \
    --eval_interval 1 \
    --save_interval 1000 \
    --save_retain_interval 5000 \
    --max_length 8192 \
    --num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --attention_backend flash \
    --overlap_param_gather true \
    --log_interval 1 \
    --overlap_grad_reduce true
