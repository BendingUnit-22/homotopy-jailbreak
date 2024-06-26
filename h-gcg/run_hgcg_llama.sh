DEVICE=0
FINETUNED_MODEL_PATH=../adapter/llama_1
BASE_MODEL_PATH=../models/Llama-2-7b-chat-hf
JUDGE_PATH=../models/HarmBench-Llama-2-13b-cls
END=1

CUDA_VISIBLE_DEVICES=${DEVICE} python gcg_llama_dynamic_budget.py --config ../configs/llama/llama_ckpt_600_cold_start.json --finetuned_model_path ${FINETUNED_MODEL_PATH}/checkpoint-600 --base_model_path ${BASE_MODEL_PATH} --judge_path ${JUDGE_PATH} --dataset_end_idx ${END}
CUDA_VISIBLE_DEVICES=${DEVICE} python gcg_llama_dynamic_budget.py --config ../configs/llama/llama_ckpt_200_warm_start.json --finetuned_model_path ${FINETUNED_MODEL_PATH}/checkpoint-200 --base_model_path ${BASE_MODEL_PATH} --judge_path ${JUDGE_PATH} --dataset_end_idx ${END}
CUDA_VISIBLE_DEVICES=${DEVICE} python gcg_llama_dynamic_budget.py --config ../configs/llama/llama_ckpt_100_warm_start.json --finetuned_model_path ${FINETUNED_MODEL_PATH}/checkpoint-100 --base_model_path ${BASE_MODEL_PATH} --judge_path ${JUDGE_PATH} --dataset_end_idx ${END}
CUDA_VISIBLE_DEVICES=${DEVICE} python gcg_llama_dynamic_budget.py --config ../configs/llama/llama_ckpt_40_warm_start.json --finetuned_model_path ${FINETUNED_MODEL_PATH}/checkpoint-40 --base_model_path ${BASE_MODEL_PATH} --judge_path ${JUDGE_PATH} --dataset_end_idx ${END}
CUDA_VISIBLE_DEVICES=${DEVICE} python gcg_llama_dynamic_budget.py --config ../configs/llama/llama_base_warm_start.json --finetuned_model_path ${BASE_MODEL_PATH} --base_model_path ${BASE_MODEL_PATH} --judge_path ${JUDGE_PATH} --dataset_end_idx ${END}