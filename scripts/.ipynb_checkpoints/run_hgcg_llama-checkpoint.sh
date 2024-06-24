DEVICE=0
CUDA_VISIBLE_DEVICES=${DEVICE} python gcg_llama_dynamic_budget.py --config ../configs/llama/llama_ckpt_600_cold_start.json
CUDA_VISIBLE_DEVICES=${DEVICE} python gcg_llama_dynamic_budget.py --config ../configs/llama/llama_ckpt_200_warm_start.json
CUDA_VISIBLE_DEVICES=${DEVICE} python gcg_llama_dynamic_budget.py --config ../configs/llama/llama_ckpt_100_warm_start.json
CUDA_VISIBLE_DEVICES=${DEVICE} python gcg_llama_dynamic_budget.py --config ../configs/llama/llama_ckpt_40_warm_start.json
CUDA_VISIBLE_DEVICES=${DEVICE} python gcg_llama_dynamic_budget.py --config ../configs/llama/llama_base_warm_start.json