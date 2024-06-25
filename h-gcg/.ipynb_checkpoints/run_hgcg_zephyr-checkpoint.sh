DEVICE=0
FINETUNED_MODEL_PATH=../models/ft_1_all_lora_zep
BASE_MODEL_PATH=../models/zephyr_7b_r2d2
JUDGE_PATH=../models/HarmBench-Llama-2-13b-cls
END=1

CUDA_VISIBLE_DEVICES=${DEVICE} python gcg_zephyr_dynamic_budget.py --config ../configs/zephyr/zephyr_ckpt_600_cold_start.json--finetuned_model_path ${FINETUNED_MODEL_PATH}/checkpoint-600 --base_model_path ${BASE_MODEL_PATH} --judge_path ${JUDGE_PATH} --dataset_end_idx ${END}
CUDA_VISIBLE_DEVICES=${DEVICE} python gcg_zephyr_dynamic_budget.py --config ../configs/zephyr/zephyr_ckpt_200_warm_start.json --finetuned_model_path ${FINETUNED_MODEL_PATH}/checkpoint-200 --base_model_path ${BASE_MODEL_PATH} --judge_path ${JUDGE_PATH} --dataset_end_idx ${END}
CUDA_VISIBLE_DEVICES=${DEVICE} python gcg_zephyr_dynamic_budget.py --config ../configs/zephyr/zephyr_ckpt_100_warm_start.json --finetuned_model_path ${FINETUNED_MODEL_PATH}/checkpoint-100 --base_model_path ${BASE_MODEL_PATH} --judge_path ${JUDGE_PATH} --dataset_end_idx ${END}
CUDA_VISIBLE_DEVICES=${DEVICE} python gcg_zephyr_dynamic_budget.py --config ../configs/zephyr/zephyr_ckpt_40_warm_start.json --finetuned_model_path ${FINETUNED_MODEL_PATH}/checkpoint-40 --base_model_path ${BASE_MODEL_PATH} --judge_path ${JUDGE_PATH} --dataset_end_idx ${END}
CUDA_VISIBLE_DEVICES=${DEVICE} python gcg_zephyr_dynamic_budget.py --config ../configs/zephyr/zephyr_base_warm_start.json --finetuned_model_path ${BASE_MODEL_PATH} --base_model_path ${BASE_MODEL_PATH} --judge_path ${JUDGE_PATH} --dataset_end_idx ${END}