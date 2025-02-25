#!/bin/bash
#*************************************************************************
# GRPO Multi-GPU Training Script for MS-Swift (RLHF)
#
# IMPORTANT CONFIGURATION:
#   The following three parameters MUST BE SET TO THE SAME VALUE:
#     --per_device_train_batch_size
#     --per_device_eval_batch_size
#     --num_generations
#
# This is required by GRPO (Generative Reinforcement Policy Optimization) to
# ensure that the global training batch size, the global evaluation batch size,
# and the number of generations per prompt are fully consistent.
#
# COMMAND LINE SWITCHES (per Swift v3 Documentation):
#
#   --rlhf_type <TYPE>
#       Specifies the RLHF algorithm to use. For GRPO training, use "grpo".
#
#   --model <MODEL_IDENTIFIER>
#       Identifier or path for the model to be fine-tuned.
#       Example: "Qwen/Qwen2.5-7B-Instruct"
#
#   --reward_funcs <FUNC_LIST>
#       Space-separated list of reward functions applied during training.
#       For instance: "accuracy format cosine repetition"
#
#   --use_vllm <BOOLEAN>
#       Flag indicating whether to use the vLLM library for accelerated inference.
#       (true/false)
#
#   --vllm_device <DEVICE>
#       Specifies which device vLLM should use. ("auto" detects automatically.)
#
#   --vllm_gpu_memory_utilization <FLOAT>
#       Fraction of the GPU memory (e.g., 0.7) that vLLM is allowed to use.
#       NOTE: THIS CAN WASTE A TON OF GPU MEMORY IF NOT SET PROPERLY.
#       Recommendation: Set to 0.5 for 8 GPUs, 0.6 for 16 GPUs, 0.7 for 32 GPUs.
#
#   --vllm_max_model_len <INT>
#       Maximum sequence length for vLLM processing.
#
#   --train_type <TYPE>
#       Describes the type of training to perform. "full" implies full
#       fine-tuning of all model parameters.
#
#   --torch_dtype <DTYPE>
#       Specifies the PyTorch data type. For example, "bfloat16" is used to
#       reduce memory consumption while maintaining numerical range.
#
#   --dataset <DATASET_IDENTIFIER>
#       The dataset to be used for training.
#       Example: "open-r1/OpenR1-Math-220k"
#
#   --max_completion_length <INT>
#       Maximum length allowed for generated completions.
#
#   --num_train_epochs <INT>
#       The number of training epochs to run.
#
#   --per_device_train_batch_size <INT>
#       Training batch size per device. MUST be equal to the eval batch size and
#       the number of generations.
#
#   --per_device_eval_batch_size <INT>
#       Evaluation batch size per device. Also must be the same value as the other two.
#
#   --learning_rate <FLOAT>
#       The learning rate for the optimizer.
#
#   --gradient_accumulation_steps <INT>
#       Number of steps to accumulate gradients before performing a model update.
#
#   --eval_steps <INT>
#       Interval (in steps) at which evaluation is performed.
#
#   --save_steps <INT>
#       Interval (in steps) at which model checkpoints are saved.
#
#   --save_total_limit <INT>
#       Maximum number of checkpoints to store.
#
#   --logging_steps <INT>
#       Interval (in steps) for logging training progress.
#
#   --max_length <INT>
#       Maximum sequence length of inputs/model context.
#
#   --output_dir <PATH>
#       Directory to save model checkpoints, logs, and outputs.
#
#   --warmup_ratio <FLOAT>
#       Proportion of total steps used for the learning rate warmup phase.
#
#   --dataloader_num_workers <INT>
#       Number of subprocesses to use for data loading.
#
#   --dataset_num_proc <INT>
#       Number of processes to use for dataset processing.
#
#   --num_generations <INT>
#       Number of generations per prompt in GRPO. MUST match the per-device batch sizes.
#
#   --temperature <FLOAT>
#       Sampling temperature used during generation (controls randomness).
#
#   --system <FILE_PATH>
#       Path to the file containing system prompts.
#
#   --deepspeed <CONFIG>
#       Specifies the DeepSpeed optimization stage to use (e.g., "zero2" or "zero3").
#
#   --log_completions <BOOLEAN>
#       Enables logging of generated completions if set to true.
# 
#   Optimizer Options:
#       --optim <OPTIMIZER>
#       Specifies the optimizer to use.
#       For instance: "adamw_torch"
#       [--optim {adamw_hf,adamw_torch,adamw_torch_fused,adamw_torch_xla,
#       adamw_torch_npu_fused,adamw_apex_fused,adafactor,adamw_anyprecision,
#       adamw_torch_4bit,adamw_torch_8bit,ademamix,sgd,adagrad,adamw_bnb_8bit,
#       adamw_8bit,ademamix_8bit,lion_8bit,lion_32bit,paged_adamw_32bit,paged_adamw_8bit,
#       paged_ademamix_32bit,paged_ademamix_8bit,paged_lion_32bit,paged_lion_8bit,
#       rmsprop,rmsprop_bnb,rmsprop_bnb_8bit,rmsprop_bnb_32bit,galore_adamw,
#       galore_adamw_8bit,galore_adafactor,galore_adamw_layerwise,
#       galore_adamw_8bit_layerwise,galore_adafactor_layerwise,lomo,adalomo,
#       grokadamw,schedule_free_radam,schedule_free_adamw,schedule_free_sgd,
#       apollo_adamw,apollo_adamw_layerwise}]
#
#   --deepspeed <CONFIG>
#       deepspeed: Defaults to None. It can be set to 'zero0', 'zero1', 'zero2',
#       'zero3', 'zero2_offload', 'zero3_offload' to use the built-in deepspeed configuration file of ms-swift.
#
# For detailed information, refer to Swift v3's documentation at:
#   https://swift.readthedocs.io/en/latest/
#
#*************************************************************************

# Detect the number of GPUs available using nvidia-smi (capped at 10)
num_gpus=$(nvidia-smi -L | wc -l)
if [ "$num_gpus" -gt 10 ]; then
  num_gpus=10
fi

# Create a comma-separated list (e.g., "0,1,2,...") for CUDA_VISIBLE_DEVICES
gpu_list=$(seq 0 $((num_gpus - 1)) | paste -sd, -)

# Set NPROC_PER_NODE to be one less than the total GPUs
nproc_per_node=$num_gpus

echo "Detected GPUs: $gpu_list"
echo "Setting NPROC_PER_NODE to: $nproc_per_node"

CUDA_VISIBLE_DEVICES=$gpu_list \
python3 -m torch.distributed.run --nproc_per_node=$nproc_per_node \
-m swift.cli.rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --reward_funcs accuracy format cosine repetition\
    --train_type lora \
    --lora_rank 4 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --optim adamw_8bit \
    --torch_dtype bfloat16 \
    --dataset 'AI-MO/NuminaMath-TIR#5000' \
    --max_completion_length 2048 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --num_generations 2 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 2 \
    --eval_steps 20 \
    --save_steps 20 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 2 \
    --temperature 0.9 \
    --system 'prompt.txt' \
    --deepspeed zero3_offload \
    --ds3_gather_for_generation false \
    --log_completions true