DEVICE=0
CUDA_VISIBLE_DEVICES=${DEVICE} python gcg_zephyr_dynamic_budget.py --config ../configs/zephyr/zephyr_ckpt_600_cold_start.json
CUDA_VISIBLE_DEVICES=${DEVICE} python gcg_zephyr_dynamic_budget.py --config ../configs/zephyr/zephyr_ckpt_200_warm_start.json
CUDA_VISIBLE_DEVICES=${DEVICE} python gcg_zephyr_dynamic_budget.py --config ../configs/zephyr/zephyr_ckpt_100_warm_start.json
CUDA_VISIBLE_DEVICES=${DEVICE} python gcg_zephyr_dynamic_budget.py --config ../configs/zephyr/zephyr_ckpt_40_warm_start.json
CUDA_VISIBLE_DEVICES=${DEVICE} python gcg_zephyr_dynamic_budget.py --config ../configs/zephyr/zephyr_base_warm_start.json