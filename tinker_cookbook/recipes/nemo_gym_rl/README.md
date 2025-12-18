# Post-training with Tinker + Nemo Gym

Train language model agents in NeMo environments using Tinker's RL training infrastructure.

**Multi-turn representation:**
- Each turn is a separate `Transition`
- Turn 1: `ob1` (prompt) → `ac1` (generation) → reward=0
- Turn 2: `ob2` (prompt + tool result) → `ac2` (generation) → reward=0
- Turn N: `obN` (full context) → `acN` (generation) → reward=final_reward


## Setup

### 1. Install Dependencies

```bash
git clone https://github.com/NVIDIA-NeMo/Gym.git
cd Gym
uv venv
source .venv/bin/activate
uv sync

cd ../tinker-cookbook
uv pip install -e ".[nemo-gym]"
```

### 2. Start Nemo Gym Servers

NeMo Gym uses `ng_run` to start all servers (head, model, resources, agent) from config files.

**On-Policy Training**: Tinker automatically serves its inference endpoint during training. The nemo gym agent will call this endpoint to generate rollouts using the model being trained (on-policy).

Create `Gym/configs/tinker_model.yaml`:
```yaml
policy_model:
  responses_api_models:
    vllm_model:
      entrypoint: app.py
      base_url: http://localhost:8000
      api_key: tinker
      model: tinker
      return_token_id_information: true
      uses_reasoning_parser: false
```

Start nemo gym servers (Terminal 1):

```bash
cd Gym

# Example: reasoning_gym with Tinker
ng_run "+config_paths=[resources_servers/reasoning_gym/configs/reasoning_gym.yaml,configs/tinker_model.yaml]"
```

You should see 4 servers running:
- Head server (port 11000)
- Resources server (e.g., port 8002)
- Agent server (e.g., port 8001)
- Model server (e.g., port 8003) - routes requests to Tinker's endpoint

**Architecture**:
### 4. Prepare Dataset

Create a JSONL file with nemo gym format:

```json
{
  "responses_create_params": {
    "input": [
      {"role": "system", "content": "You are a helpful assistant..."},
      {"role": "user", "content": "What is the weather in San Francisco?"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get weather for a location",
          "parameters": {...}
        }
      }
    ]
  },
  "expected_answers": ["sunny", "75F"],
  "metadata": {"task": "weather_query", "difficulty": "easy"},
  "ground_truth": {"temperature": 75, "condition": "sunny"}
}
```

**Field descriptions:**
- `responses_create_params`: Sent to the model server. Contains the initial prompt and available tools.
- `expected_answers`, `metadata`, `ground_truth`: Sent to the resources server's `/verify` endpoint to compute rewards.

**How it works:**

When the Tinker training script calls the agent's `/run` endpoint:

1. Agent receives the entire dataset item
2. Agent calls resources server `/seed_session` with the full item (initializes episode state)
3. Agent orchestrates multi-turn loop:
   - Model generates (using `responses_create_params`)
   - If model calls tools, agent forwards to resources server
   - Resources server returns tool results
   - Process repeats until model stops or max_steps reached
4. Agent calls resources server `/verify` with the trajectory + expected_answers/ground_truth
5. Resources server computes reward (e.g., checking if answer matches expected_answers)
6. Agent returns `{reward: float, response: {output: [...]}}` to Tinker
7. Tinker extracts token IDs and trains on the trajectory

## Usage

### Training

```bash
python -m tinker_cookbook.recipes.nemo_gym_rl.train \
    dataset_path=/path/to/dataset.jsonl \
    head_server_host=127.0.0.1 \
    head_server_port=11000 \
    agent_name=simple_agent \
    model_name=Qwen/Qwen3-4B-Instruct-2507
```

### Evaluation

```bash
python -m tinker_cookbook.recipes.nemo_gym_rl.evaluate \
    dataset_path=/path/to/dataset.jsonl \
    agent_server=http://127.0.0.1:8001 \
    num_examples=10 \
    rollouts_per_example=3 \
    max_output_tokens=4096 \
    temperature=1.0
```


## Output

Training produces:
- **Checkpoints**: Saved in `{log_path}/checkpoints/`
- **Trajectories**: Logged to `{log_path}/trajectories.jsonl`
- **Metrics**: Training metrics in logs and W&B (if configured)

Trajectory format:
```json
{
  "step": 0,
  "rollout_idx": 0,
  "reward": 0.85,
  "output": [
    {
      "type": "message",
      "prompt_token_ids": [...],
      "generation_token_ids": [...],
      "generation_log_probs": [...]
    },
    ...
  ],
  "error": null
}
```

## Reasoning Gym Example

### 1. Create dataset

```bash
cd Gym/resources_servers/reasoning_gym

python scripts/create_dataset.py \
    --task knights_knaves \
    --size 500 \
    --seed 42 \
    --output data/train_knights_knaves.jsonl
```

### 2. Create Tinker model config

```bash
cd Gym

cat > configs/tinker_model.yaml << 'EOF'
policy_model:
  responses_api_models:
    vllm_model:
      entrypoint: app.py
      base_url: http://localhost:8000
      api_key: tinker
      model: tinker
      return_token_id_information: true
      uses_reasoning_parser: false
EOF
```

### 3. Start nemo gym servers (Terminal 1)

```bash
cd Gym

ng_run "+config_paths=[resources_servers/reasoning_gym/configs/reasoning_gym.yaml,configs/tinker_model.yaml]"
```

### 4. Train with Tinker (Terminal 2)

```bash
python -m tinker_cookbook.recipes.nemo_gym_rl.train \
    dataset_path=../Gym/resources_servers/reasoning_gym/data/train_knights_knaves.jsonl \
    head_server_host=127.0.0.1 \
    head_server_port=11000 \
    model_name=meta-llama/Llama-3.1-8B-Instruct \
    group_size=4 \
    groups_per_batch=16 \
    learning_rate=1e-5 \
    lora_rank=32 \
    max_tokens=2048 \
    save_every=10 \
    log_path=./outputs/reasoning_gym_knights
```

### 5. Evaluate

```bash
python -m tinker_cookbook.recipes.nemo_gym_rl.evaluate \
    dataset_path=Gym/resources_servers/reasoning_gym/data/train_knights_knaves.jsonl \
    num_examples=20 \
    rollouts_per_example=3
```


Notes: 

Nemo gym agent returns:
```python
{
    "reward": float,  # From /verify
    "response": {
        "output": [
            {
                "type": "message" | "reasoning" | "function_call" | "function_call_output",
                "prompt_token_ids": List[int],  # Full context
                "generation_token_ids": List[int],  # This turn's generation
                "generation_log_probs": List[float],
                ...
            }
        ]
    }
}
```

### Conversion Logic

Each output item with token information becomes a `Transition`:
- `ob` = `ModelInput.from_ints(prompt_token_ids)` - Full context including tool results
- `ac` = `TokensWithLogprobs(generation_token_ids, generation_log_probs)` - This turn's action
- `reward` = 0.0 (except final turn gets episode reward)
- `episode_done` = True only on last turn
