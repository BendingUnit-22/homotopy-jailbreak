# 600->200->100->40->base
DEVICE=3
TEMPLATE=llama2
START=0
END=1
STEPS=40


### Cold start, 600
MODEL=../adapter/llama_1/checkpoint-600
SAVE_NAME=ft_1_all_lora_llama_cold_start_40_iter

CUDA_VISIBLE_DEVICES=${DEVICE} python3 autodan_hga_eval_llama_lora_tiered_dynamic_budgeting.py \
    --model ${MODEL} \
    --save_name ${SAVE_NAME} \
    --template ${TEMPLATE} \
    --start ${START} \
    --end ${END} \
    --log_interval 5 \
    --num_steps ${STEPS} \


### 600->200
MODEL=../models/ft_1_all_lora_llama/checkpoint-200
SAVE_NAME=ft_1_all_lora_llama_warm_start_40_iter_from_600
WARM_START_PATH=../data/llama2/autodan_checkpoint-600_ft_1_all_lora_llama_cold_start_40_iter_advbench_0_520_start_0_end_1.json

CUDA_VISIBLE_DEVICES=${DEVICE} python3 autodan_hga_eval_llama_lora_tiered_dynamic_budgeting.py \
    --model ${MODEL} \
    --save_name ${SAVE_NAME} \
    --template ${TEMPLATE} \
    --start ${START} \
    --end ${END} \
    --log_interval 5 \
    --warm_start True \
    --warm_start_path ${WARM_START_PATH} \
    --num_steps ${STEPS} \
  

### 200->100
MODEL=../models/ft_1_all_lora_llama/checkpoint-100
SAVE_NAME=ft_1_all_lora_llama_warm_start_40_iter_from_200
WARM_START_PATH=../data/llama2/autodan_checkpoint-200_ft_1_all_lora_llama_warm_start_40_iter_from_600_advbench_0_520_start_0_end_1.json

CUDA_VISIBLE_DEVICES=${DEVICE} python3 autodan_hga_eval_llama_lora_tiered_dynamic_budgeting.py \
    --model ${MODEL} \
    --save_name ${SAVE_NAME} \
    --template ${TEMPLATE} \
    --start ${START} \
    --end ${END} \
    --log_interval 5 \
    --warm_start True \
    --warm_start_path ${WARM_START_PATH} \
    --num_steps ${STEPS} \


### 100->40
MODEL=../models/ft_1_all_lora_llama/checkpoint-40
SAVE_NAME=ft_1_all_lora_llama_warm_start_40_iter_from_100
WARM_START_PATH=../data/llama2/autodan_checkpoint-100_ft_1_all_lora_llama_warm_start_40_iter_from_200_advbench_0_520_start_0_end_1.json

CUDA_VISIBLE_DEVICES=${DEVICE} python3 autodan_hga_eval_llama_lora_tiered_dynamic_budgeting.py \
    --model ${MODEL} \
    --save_name ${SAVE_NAME} \
    --template ${TEMPLATE} \
    --start ${START} \
    --end ${END} \
    --log_interval 5 \
    --warm_start True \
    --warm_start_path ${WARM_START_PATH} \
    --num_steps ${STEPS} \
    
    
### 40->base
MODEL=/nobackup4/hooda/models/Llama-2-7b-chat-hf
SAVE_NAME=warm_start_40_iter_from_40
WARM_START_PATH=../data/llama2/autodan_checkpoint-40_ft_1_all_lora_llama_warm_start_40_iter_from_100_advbench_0_520_start_0_end_1.json

CUDA_VISIBLE_DEVICES=${DEVICE} python3 autodan_hga_eval_llama_lora_tiered_dynamic_budgeting.py \
    --model ${MODEL} \
    --save_name ${SAVE_NAME} \
    --template ${TEMPLATE} \
    --start ${START} \
    --end ${END} \
    --log_interval 5 \
    --warm_start True \
    --warm_start_path ${WARM_START_PATH} \
    --num_steps ${STEPS} \