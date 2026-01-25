#!/bin/bash
set -x

export RAY_SERVE_QUEUE_LENGTH_RESPONSE_DEADLINE_S=10.0

ray start --head

serve run remote_reward_server:app --name evaluator &

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=data/train.parquet \
    data.val_files=data/test.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=6144 \
    data.truncation="right" \
    actor_rollout_ref.model.path=deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0001 \
    actor_rollout_ref.actor.kl_loss_type=mse \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.kl_ctrl.kl_coef=0.0001 \
    custom_reward_function.path=code_reward_func_think.py \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    trainer.project_name='code-mtrl' \
    trainer.experiment_name="r1-llama-8b-taco-single-turn-grpo-$(date +%Y%m%d-%H%M%S)" \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=4 \
    trainer.default_local_dir=~/r1-llama-8b-taco-single-turn-grpo \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=4 \
    reward_model.reward_manager=prime $@ 2>&1 | tee grpo_r1_llama.log
    
# bash train_grpo_single_turn.sh