# homotopy-jailbreak
This repo contains the evaluation for an anon EMNLP submission.

## Prerequisite
We first create a conda environment and then install some necessary packages:

conda create -n homotopy python=3.11.8

conda activate homotopy

pip install -r requirements.txt

## Model fine tuning
We use [huggingface SFT Trainer](https://huggingface.co/docs/trl/en/sft_trainer) to fine tune the model.

To fine-tune the model, one can run

```python ft.py --model_path LLAMA_PATH --model llama```

The base model can be either [Llama-2](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/tree/main) or [zephyr_7b_r2d2](https://huggingface.co/cais/zephyr_7b_r2d2).

We use ```--model_path``` to specify where the base model is stored and ```--model``` (the option is either ```llama``` or ```zep```) to specify the type of the model. 

One can also change the fine-tuning batch size using flag ```--batch_size``` and also ```--epochs``` to specify how many epochs are desired for the fine-tuning. 

```--check_steps``` is used to specify how frequently the model checkpoint is saved. In this project, we use LoRA fine-tuning to tune the base model. By default, the LoRA checkpoint adapters are saved in ```./adapter``` directory.
