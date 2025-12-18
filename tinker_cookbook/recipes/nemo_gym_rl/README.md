# Post-training with Tinker + Nemo Gym

Train models with NeMo Gym agents and environments using Tinker's managed training service.

### Install Dependencies

```bash
git clone https://github.com/NVIDIA-NeMo/Gym.git
cd Gym
uv venv
source .venv/bin/activate
uv sync

cd ../tinker-cookbook
uv pip install -e ".[nemo-gym]"
```

### Start Nemo Gym Servers

NeMo Gym uses `ng_run` to start all servers (head, model, resources, agent) from config files.

Create vllm_model or use env.yaml:
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

Start nemo gym servers:

```bash
cd Gym

ng_run "+config_paths=[resources_servers/workplace_assistant/configs/workplace_assistant.yaml,configs/tinker_model.yaml]"
```


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

### Output

- **Checkpoints**: Saved in `{log_path}/checkpoints/`
- **Trajectories**: Logged to `{log_path}/trajectories.jsonl`
- **Metrics**: Training metrics in logs and W&B (if configured)