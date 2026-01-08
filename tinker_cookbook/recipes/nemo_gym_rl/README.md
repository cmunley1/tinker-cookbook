# Post-training with Tinker + Nemo Gym

Train agents with NeMo Gym using Tinker's managed training service.

First time setup
```bash
git clone https://github.com/NVIDIA-NeMo/Gym.git
cd Gym
uv venv
source .venv/bin/activate
uv sync

cd ..
git clone https://github.com/cmunley1/tinker-cookbook.git
git checkout cmunley1/nemo_gym
uv pip install -e ".[nemo-gym]"
```

Regular setup
```bash
source Gym/.venv/bin/activate
cd tinker-cookbook
export TINKER_API_KEY=tml-...
```

Launch Nemo Gym server manually (can be automated later on):
```bash
ng_run "+config_paths=[resources_servers/workplace_assistant/configs/workplace_assistant.yaml,configs/tinker_model.yaml]"
```

Note that tinker_model.yaml is not yet pushed to NeMo-Gym, but should look like this: 
```bash
cat ~/Gym/configs/tinker_model.yaml 
policy_model:
  responses_api_models:
    vllm_model:
      entrypoint: app.py
      base_url: http://localhost:8000/v1
      api_key: tinker
      model: tinker
      return_token_id_information: true
      uses_reasoning_parser: false
```


Start training
```bash
python -m tinker_cookbook.recipes.nemo_gym_rl.train     dataset_path=/home/cmunley/tinker-cookbook/train-workplace.jsonl     model_name=Qwen/Qwen3-4B-Instruct-2507     group_size=8     groups_per_batch=8     learning_rate=1e-5     lora_rank=32     max_tokens=8192     save_every=10     log_path=./outputs/qwen3_4bI_workplace_run1
```


Plot train reward 
```bash
python3 plot_rewards.py --log_path outputs/workplace_run1/
```