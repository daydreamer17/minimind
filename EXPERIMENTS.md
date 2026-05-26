# MiniMind 实验与运行命令手册

本文档同步当前仓库的主要训练、评测、服务脚本，整理各阶段依赖关系和常用命令模板。命令以 `/root/minimind` 这个工作区为准。

## 使用前说明

建议从 `trainer/` 目录启动训练脚本：

```bash
cd /root/minimind/trainer
```

训练脚本里大量路径是相对路径，例如 `../out`、`../checkpoints`。从项目根目录直接运行训练脚本，保存和加载路径可能会跑偏。

评测和服务脚本的推荐运行目录不同：

- `eval_llm.py`：从项目根目录 `/root/minimind` 运行
- `scripts/*.py`：从 `/root/minimind/scripts` 运行

常见数据和模型路径占位符：

- `/root/autodl-tmp/dir/pretrain_hq.jsonl`
- `/root/autodl-tmp/dir/sft_mini_512.jsonl`
- `/root/autodl-tmp/dir/lora_identity.jsonl`
- `/root/autodl-tmp/dir/dpo.jsonl`
- `/root/autodl-tmp/dir/r1_mix_1024.jsonl`
- `/root/autodl-tmp/dir/rlaif-mini.jsonl`
- `/root/autodl-tmp/dir/agent_rl.jsonl`
- `/root/autodl-tmp/dir/agent_rl_math.jsonl`
- `/root/autodl-tmp/internlm2-1_8b-reward`

断点续训和“基于旧权重开新实验”不是一回事：

- 精确续训：`--from_resume 1`
- 从旧权重重新开实验：`--from_resume 0`，并设置 `--from_weight`

多卡训练把 `python` 换成 `torchrun`：

```bash
torchrun --nproc_per_node=2 train_xxx.py ...
```

训练监控参数仍叫 `--use_wandb` / `--wandb_project`。当前大部分脚本仍直接 `import wandb`，`train_agent.py` 使用 `swanlab as wandb`，因此具体登录方式取决于你实际安装和使用的后端。

## 实验依赖关系

当前主线推荐：

```text
Tokenizer(通常不重训)
  -> Pretrain
  -> Full SFT
  -> DPO / RLAIF(GRPO/CISPO/PPO) / Agentic RL
  -> Distillation(可选)

Full SFT
  -> LoRA
```

说明：

- `toolcall` 数据已经混入当前 SFT 数据，通常不再需要单独 Tool Calling SFT。
- README 中当前主线不再强调单独维护 `reason_*.pth`；但仓库里仍有 `train_reason.py`，可作为可选/兼容实验使用。
- `train_grpo.py` 已支持 `--loss_type grpo|cispo`，默认是 `cispo`。
- `train_agent.py` 面向多轮 Tool-Use / Agentic RL，也支持 `--loss_type grpo|cispo`，默认是 `cispo`。
- `train_ppo.py`、`train_grpo.py` 现在通过 `--from_weight` 指定基座权重，不再使用旧版 `--reasoning` 参数。
- `train_spo.py` 仍保留旧式 `--reasoning` 参数，`--reasoning 1` 基于 `reason`，`--reasoning 0` 基于 `full_sft`。

## 权重命名与产物位置

普通训练产物：

```text
/root/minimind/out/{save_weight}_{hidden_size}.pth
/root/minimind/out/{save_weight}_{hidden_size}_moe.pth
```

续训状态：

```text
/root/minimind/checkpoints/{save_weight}_{hidden_size}_resume.pth
/root/minimind/checkpoints/{save_weight}_{hidden_size}_moe_resume.pth
```

LoRA 产物：

```text
/root/minimind/out/lora/{lora_name}_{hidden_size}.pth
```

精确续训时，下面参数要和原实验保持一致：

- `save_weight`
- `hidden_size`
- `num_hidden_layers`
- `use_moe`
- 对蒸馏脚本还要保持 `student_*` / `teacher_*` 结构参数一致

## 1. Tokenizer 实验

脚本：`train_tokenizer.py`

用途是训练 BPE + ByteLevel tokenizer，并生成包含 `<tool_call>`、`<tool_response>`、`<think>` 等模板标记的 tokenizer 配置。项目通常不建议重复训练 tokenizer；这个脚本主要用于学习 tokenizer 构建流程。

```bash
cd /root/minimind/trainer
python train_tokenizer.py
```

如果要改路径或词表大小，直接修改脚本里的常量：

- `DATA_PATH`
- `TOKENIZER_DIR`
- `VOCAB_SIZE`

## 2. Pretrain 实验

脚本：`train_pretrain.py`

用途是从头或从旧权重继续做基础语言模型预训练。

```bash
cd /root/minimind/trainer
python train_pretrain.py \
  --save_weight pretrain \
  --epochs 2 \
  --batch_size 32 \
  --accumulation_steps 8 \
  --learning_rate 5e-4 \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --max_seq_len 340 \
  --use_moe 0 \
  --data_path /root/autodl-tmp/dir/pretrain_hq.jsonl \
  --from_weight none \
  --use_wandb \
  --wandb_project MiniMind-Pretrain
```

MoE 预训练示例：

```bash
cd /root/minimind/trainer
python train_pretrain.py \
  --save_weight pretrain \
  --hidden_size 640 \
  --num_hidden_layers 8 \
  --use_moe 1 \
  --data_path /root/autodl-tmp/dir/pretrain_hq.jsonl \
  --use_wandb \
  --wandb_project MiniMind-Pretrain-MoE
```

断点续训：

```bash
cd /root/minimind/trainer
python train_pretrain.py \
  --save_weight pretrain \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --use_moe 0 \
  --data_path /root/autodl-tmp/dir/pretrain_hq.jsonl \
  --from_resume 1 \
  --use_wandb \
  --wandb_project MiniMind-Pretrain
```

## 3. Full SFT 实验

脚本：`train_full_sft.py`

用途是基于 `pretrain` 权重做全参数监督微调。当前主线 SFT 数据已经混入基础 Tool Calling 样本。

```bash
cd /root/minimind/trainer
python train_full_sft.py \
  --save_weight full_sft \
  --epochs 2 \
  --batch_size 16 \
  --learning_rate 1e-5 \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --max_seq_len 768 \
  --use_moe 0 \
  --data_path /root/autodl-tmp/dir/sft_mini_512.jsonl \
  --from_weight pretrain \
  --use_wandb \
  --wandb_project MiniMind-Full-SFT
```

断点续训：

```bash
cd /root/minimind/trainer
python train_full_sft.py \
  --save_weight full_sft \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --use_moe 0 \
  --data_path /root/autodl-tmp/dir/sft_mini_512.jsonl \
  --from_resume 1 \
  --use_wandb \
  --wandb_project MiniMind-Full-SFT
```

## 4. LoRA 实验

脚本：`train_lora.py`

用途是基于 `full_sft` 权重做 LoRA 微调，默认保存到 `../out/lora/`。

```bash
cd /root/minimind/trainer
python train_lora.py \
  --lora_name lora_identity \
  --epochs 10 \
  --batch_size 32 \
  --learning_rate 1e-4 \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --max_seq_len 340 \
  --use_moe 0 \
  --data_path /root/autodl-tmp/dir/lora_identity.jsonl \
  --from_weight full_sft \
  --use_wandb \
  --wandb_project MiniMind-LoRA
```

断点续训：

```bash
cd /root/minimind/trainer
python train_lora.py \
  --lora_name lora_identity \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --use_moe 0 \
  --data_path /root/autodl-tmp/dir/lora_identity.jsonl \
  --from_resume 1 \
  --use_wandb \
  --wandb_project MiniMind-LoRA
```

## 5. DPO 实验

脚本：`train_dpo.py`

用途是基于 `full_sft` 做偏好优化。

```bash
cd /root/minimind/trainer
python train_dpo.py \
  --save_weight dpo \
  --epochs 1 \
  --batch_size 4 \
  --learning_rate 4e-8 \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --max_seq_len 1024 \
  --use_moe 0 \
  --data_path /root/autodl-tmp/dir/dpo.jsonl \
  --from_weight full_sft \
  --beta 0.15 \
  --use_wandb \
  --wandb_project MiniMind-DPO
```

断点续训：

```bash
cd /root/minimind/trainer
python train_dpo.py \
  --save_weight dpo \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --use_moe 0 \
  --data_path /root/autodl-tmp/dir/dpo.jsonl \
  --from_resume 1 \
  --use_wandb \
  --wandb_project MiniMind-DPO
```

## 6. Reasoning 兼容实验

脚本：`train_reason.py`

README 当前主线说明不再单独维护 `reason_*.pth`，但脚本仍存在。需要复现旧链路或单独做 reasoning 格式对齐时，可以继续使用它。

```bash
cd /root/minimind/trainer
python train_reason.py \
  --save_weight reason \
  --epochs 1 \
  --batch_size 8 \
  --learning_rate 1e-6 \
  --hidden_size 512 \
  --num_hidden_layers 8 \
  --max_seq_len 720 \
  --use_moe 0 \
  --data_path /root/autodl-tmp/dir/r1_mix_1024.jsonl \
  --from_weight dpo \
  --use_wandb \
  --wandb_project MiniMind-Reasoning
```

断点续训：

```bash
cd /root/minimind/trainer
python train_reason.py \
  --save_weight reason \
  --hidden_size 512 \
  --num_hidden_layers 8 \
  --use_moe 0 \
  --data_path /root/autodl-tmp/dir/r1_mix_1024.jsonl \
  --from_resume 1 \
  --use_wandb \
  --wandb_project MiniMind-Reasoning
```

## 7. PPO 实验

脚本：`train_ppo.py`

用途是基于 `full_sft`、`dpo` 或其他指定权重做 PPO。当前脚本通过 `--from_weight` 指定基座权重，并支持 `torch` / `sglang` rollout engine。

```bash
cd /root/minimind/trainer
python train_ppo.py \
  --save_weight ppo_actor \
  --epochs 1 \
  --batch_size 2 \
  --learning_rate 3e-7 \
  --critic_learning_rate 5e-7 \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --use_moe 0 \
  --max_seq_len 768 \
  --max_gen_len 1024 \
  --data_path /root/autodl-tmp/dir/rlaif-mini.jsonl \
  --from_weight full_sft \
  --reward_model_path /root/autodl-tmp/internlm2-1_8b-reward \
  --thinking_ratio 0.9 \
  --rollout_engine torch \
  --use_wandb \
  --wandb_project MiniMind-PPO
```

使用 SGLang rollout：

```bash
cd /root/minimind/trainer
python train_ppo.py \
  --save_weight ppo_actor \
  --from_weight full_sft \
  --data_path /root/autodl-tmp/dir/rlaif-mini.jsonl \
  --reward_model_path /root/autodl-tmp/internlm2-1_8b-reward \
  --rollout_engine sglang \
  --sglang_base_url http://localhost:8998 \
  --sglang_shared_path ./sglang_ckpt_ppo \
  --use_wandb \
  --wandb_project MiniMind-PPO
```

断点续训：

```bash
cd /root/minimind/trainer
python train_ppo.py \
  --save_weight ppo_actor \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --use_moe 0 \
  --data_path /root/autodl-tmp/dir/rlaif-mini.jsonl \
  --from_resume 1 \
  --use_wandb \
  --wandb_project MiniMind-PPO
```

## 8. GRPO / CISPO 实验

脚本：`train_grpo.py`

用途是基于指定权重做组内相对优势优化。`--loss_type cispo` 是当前默认值，切到标准 GRPO 时设置 `--loss_type grpo`。

```bash
cd /root/minimind/trainer
python train_grpo.py \
  --save_weight grpo \
  --epochs 1 \
  --batch_size 2 \
  --learning_rate 3e-7 \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --use_moe 0 \
  --max_seq_len 768 \
  --max_gen_len 1024 \
  --data_path /root/autodl-tmp/dir/rlaif-mini.jsonl \
  --from_weight full_sft \
  --num_generations 6 \
  --beta 0.1 \
  --loss_type cispo \
  --thinking_ratio 0.9 \
  --reward_model_path /root/autodl-tmp/internlm2-1_8b-reward \
  --rollout_engine torch \
  --use_wandb \
  --wandb_project MiniMind-GRPO
```

标准 GRPO：

```bash
cd /root/minimind/trainer
python train_grpo.py \
  --save_weight grpo \
  --from_weight full_sft \
  --data_path /root/autodl-tmp/dir/rlaif-mini.jsonl \
  --loss_type grpo \
  --reward_model_path /root/autodl-tmp/internlm2-1_8b-reward \
  --use_wandb \
  --wandb_project MiniMind-GRPO
```

SGLang rollout：

```bash
cd /root/minimind/trainer
python train_grpo.py \
  --save_weight grpo \
  --from_weight full_sft \
  --data_path /root/autodl-tmp/dir/rlaif-mini.jsonl \
  --rollout_engine sglang \
  --sglang_base_url http://localhost:8998 \
  --sglang_shared_path ./sglang_ckpt_grpo \
  --reward_model_path /root/autodl-tmp/internlm2-1_8b-reward \
  --use_wandb \
  --wandb_project MiniMind-GRPO
```

## 9. SPO 兼容实验

脚本：`train_spo.py`

用途是基于旧版 reasoning/full_sft 链路做 SPO。这个脚本仍使用 `--reasoning` 控制基座权重。

```bash
cd /root/minimind/trainer
python train_spo.py \
  --save_weight spo \
  --epochs 1 \
  --batch_size 2 \
  --learning_rate 1e-7 \
  --accumulation_steps 4 \
  --hidden_size 512 \
  --num_hidden_layers 8 \
  --use_moe 0 \
  --max_seq_len 66 \
  --max_gen_len 1536 \
  --data_path /root/autodl-tmp/dir/rlaif-mini.jsonl \
  --beta 0.02 \
  --reasoning 1 \
  --reward_model_path /root/autodl-tmp/internlm2-1_8b-reward \
  --use_wandb \
  --wandb_project MiniMind-SPO
```

断点续训：

```bash
cd /root/minimind/trainer
python train_spo.py \
  --save_weight spo \
  --hidden_size 512 \
  --num_hidden_layers 8 \
  --use_moe 0 \
  --data_path /root/autodl-tmp/dir/rlaif-mini.jsonl \
  --reasoning 1 \
  --from_resume 1 \
  --use_wandb \
  --wandb_project MiniMind-SPO
```

## 10. Agentic RL 实验

脚本：`train_agent.py`

用途是在多轮 Tool-Use 场景中做 Agentic RL。数据通常使用 `agent_rl.jsonl` 或 `agent_rl_math.jsonl`，样本里包含 `messages`、`tools` 和 `gt`；训练时会 rollout 多轮工具调用轨迹，再基于工具合法性、GT 命中、格式、RM 分数和未完成惩罚计算 reward。

默认 torch rollout：

```bash
cd /root/minimind/trainer
python train_agent.py \
  --save_weight agent \
  --epochs 1 \
  --batch_size 2 \
  --learning_rate 3e-7 \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --use_moe 0 \
  --max_seq_len 1024 \
  --max_gen_len 768 \
  --max_total_len 2500 \
  --data_path /root/autodl-tmp/dir/agent_rl.jsonl \
  --from_weight full_sft \
  --num_generations 4 \
  --beta 0.1 \
  --loss_type cispo \
  --thinking_ratio 0.1 \
  --reward_model_path /root/autodl-tmp/internlm2-1_8b-reward \
  --rollout_engine torch \
  --use_wandb \
  --wandb_project MiniMind-Agent-RL
```

数学 Tool-Use / RLVR 数据：

```bash
cd /root/minimind/trainer
python train_agent.py \
  --save_weight agent \
  --data_path /root/autodl-tmp/dir/agent_rl_math.jsonl \
  --from_weight full_sft \
  --loss_type cispo \
  --reward_model_path /root/autodl-tmp/internlm2-1_8b-reward \
  --use_wandb \
  --wandb_project MiniMind-Agent-RL
```

SGLang rollout：

```bash
cd /root/minimind/trainer
python train_agent.py \
  --save_weight agent \
  --from_weight full_sft \
  --data_path /root/autodl-tmp/dir/agent_rl_math.jsonl \
  --rollout_engine sglang \
  --sglang_base_url http://localhost:8998 \
  --sglang_model_path ../model \
  --sglang_shared_path ./sglang_ckpt_agent \
  --reward_model_path /root/autodl-tmp/internlm2-1_8b-reward \
  --use_wandb \
  --wandb_project MiniMind-Agent-RL
```

断点续训：

```bash
cd /root/minimind/trainer
python train_agent.py \
  --save_weight agent \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --use_moe 0 \
  --data_path /root/autodl-tmp/dir/agent_rl.jsonl \
  --from_resume 1 \
  --use_wandb \
  --wandb_project MiniMind-Agent-RL
```

调试 rollout：

```bash
cd /root/minimind/trainer
python train_agent.py \
  --data_path /root/autodl-tmp/dir/agent_rl_math.jsonl \
  --debug_mode \
  --debug_interval 1
```

## 11. Distillation 实验

脚本：`train_distillation.py`

用途是用教师模型蒸馏学生模型。当前脚本已经拆分了学生和教师的 MoE 配置，分别使用 `--student_use_moe` 和 `--teacher_use_moe`。

```bash
cd /root/minimind/trainer
python train_distillation.py \
  --save_weight full_dist \
  --epochs 6 \
  --batch_size 32 \
  --learning_rate 5e-6 \
  --max_seq_len 340 \
  --data_path /root/autodl-tmp/dir/sft_mini_512.jsonl \
  --student_hidden_size 768 \
  --student_num_layers 8 \
  --student_use_moe 0 \
  --teacher_hidden_size 768 \
  --teacher_num_layers 8 \
  --teacher_use_moe 1 \
  --from_student_weight full_sft \
  --from_teacher_weight full_sft \
  --alpha 0.5 \
  --temperature 1.5 \
  --use_wandb \
  --wandb_project MiniMind-Distillation
```

断点续训：

```bash
cd /root/minimind/trainer
python train_distillation.py \
  --save_weight full_dist \
  --student_hidden_size 768 \
  --student_num_layers 8 \
  --student_use_moe 0 \
  --teacher_hidden_size 768 \
  --teacher_num_layers 8 \
  --teacher_use_moe 1 \
  --data_path /root/autodl-tmp/dir/sft_mini_512.jsonl \
  --from_resume 1 \
  --use_wandb \
  --wandb_project MiniMind-Distillation
```

## 12. 本地推理与评测

### 12.1 通用对话评测

脚本：`eval_llm.py`

从项目根目录运行：

```bash
cd /root/minimind
python eval_llm.py \
  --weight full_sft \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --use_moe 0 \
  --max_new_tokens 512
```

预训练模型更适合续写测试，不适合直接按助手问答能力评估：

```bash
cd /root/minimind
python eval_llm.py \
  --weight pretrain \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --use_moe 0 \
  --max_new_tokens 256
```

MoE 预训练权重示例：

```bash
cd /root/minimind
python eval_llm.py \
  --weight pretrain \
  --hidden_size 640 \
  --num_hidden_layers 8 \
  --use_moe 1 \
  --max_new_tokens 256
```

Agent 权重：

```bash
cd /root/minimind
python eval_llm.py \
  --weight agent \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --use_moe 0 \
  --max_new_tokens 512 \
  --open_thinking 1
```

常用采样参数：

```bash
--temperature 0.85
--top_p 0.95
--historys 0
--show_speed 1
--open_thinking 0
--enable_thinking 0
```

### 12.2 Tool Calling 评测

脚本：`scripts/eval_toolcall.py`

本地模型：

```bash
cd /root/minimind/scripts
python eval_toolcall.py \
  --backend local \
  --weight full_sft \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --use_moe 0 \
  --max_new_tokens 512
```

Agent 权重：

```bash
cd /root/minimind/scripts
python eval_toolcall.py \
  --backend local \
  --weight agent \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --use_moe 0 \
  --max_new_tokens 512
```

OpenAI 兼容 API 后端：

```bash
cd /root/minimind/scripts
python eval_toolcall.py \
  --backend api \
  --api_base_url http://localhost:8998/v1 \
  --api_key sk-123 \
  --api_model minimind \
  --stream 1
```

### 12.3 Hugging Face / Transformers 目录

如果已经导出成 Transformers 格式，可以直接用 `--load_from`：

```bash
cd /root/minimind
python eval_llm.py \
  --load_from ./MiniMind2 \
  --max_new_tokens 512
```

### 12.4 RoPE 外推

RoPE 外推只解决位置编码范围，不代表模型真实长文本理解能力同步提升。

```bash
cd /root/minimind
python eval_llm.py \
  --weight full_sft \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --use_moe 0 \
  --inference_rope_scaling \
  --max_new_tokens 2048
```

### 12.5 榜单评测

仓库 README 推荐使用 `lm-evaluation-harness` 跑 C-Eval、CMMLU、A-CLUE、TMMLU+ 等榜单。当前仓库本身没有提供一键 benchmark 脚本。

```bash
lm_eval \
  --model hf \
  --model_args pretrained=<填写模型路径>,device=cuda,dtype=auto \
  --tasks ceval* \
  --batch_size 8 \
  --trust_remote_code
```

原生 torch 权重需要先用 `scripts/convert_model.py` 导出为 Transformers 格式。MoE 权重应走 `convert_torch2transformers_minimind`，不要走 Llama 兼容转换。

## 13. 服务与 Demo

### 13.1 OpenAI API 兼容服务

脚本：`scripts/serve_openai_api.py`

默认服务地址：

```text
http://0.0.0.0:8998/v1/chat/completions
```

启动本地 torch 权重：

```bash
cd /root/minimind/scripts
python serve_openai_api.py \
  --weight full_sft \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --use_moe 0 \
  --device cuda
```

启动 Agent 权重：

```bash
cd /root/minimind/scripts
python serve_openai_api.py \
  --weight agent \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --use_moe 0 \
  --device cuda
```

启动 Hugging Face / Transformers 目录：

```bash
cd /root/minimind/scripts
python serve_openai_api.py \
  --load_from ../MiniMind2 \
  --device cuda
```

### 13.2 命令行 API 客户端

脚本：`scripts/chat_api.py`

先启动 `serve_openai_api.py`，再运行：

```bash
cd /root/minimind/scripts
python chat_api.py
```

旧版 `chat_openai_api.py` 不再是当前 README 主推入口；如果本地还保留该文件，也可以作为简单 OpenAI 兼容客户端使用。

### 13.3 Streamlit Web Demo

脚本：`scripts/web_demo.py`

```bash
cd /root/minimind/scripts
streamlit run web_demo.py
```

Web Demo 支持本地模型模式和 API 模式。使用本仓库 `serve_openai_api.py` 时，API URL 应指向：

```text
http://127.0.0.1:8998/v1
```

## 常见链路速查

常规模型：

```text
Pretrain -> Full SFT -> DPO
```

RLAIF / CISPO：

```text
Pretrain -> Full SFT -> GRPO/CISPO/PPO
```

Tool Calling / Agent：

```text
Pretrain -> Full SFT(toolcall mixed) -> Agentic RL
```

LoRA 轻量迁移：

```text
Full SFT -> LoRA
```

蒸馏：

```text
Teacher + Student -> Distillation
```

## 续训速查

精确续训通常只需要：

1. 进入 `trainer/`
2. 使用和原实验一致的 `save_weight`
3. 使用和原实验一致的模型结构参数
4. 加上 `--from_resume 1`

例如继续 SFT：

```bash
cd /root/minimind/trainer
python train_full_sft.py \
  --save_weight full_sft \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --use_moe 0 \
  --data_path /root/autodl-tmp/dir/sft_mini_512.jsonl \
  --from_resume 1 \
  --use_wandb \
  --wandb_project MiniMind-Full-SFT
```

## 常用调试建议

先小 batch 冒烟：

```bash
python train_grpo.py \
  --batch_size 1 \
  --num_generations 2 \
  --log_interval 1 \
  --debug_mode \
  --debug_interval 1
```

RL 类训练优先观察：

- `reward`
- `kl_ref`
- `group_reward_std`
- `policy_loss`
- `avg_response_len`

Agent 类训练额外观察：

- 工具调用 JSON 是否闭合
- 工具名和参数是否在允许列表内
- `gt` 是否能在最终回答中命中
- 多轮 observation 是否被正确拼回上下文
