import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import load_dataset
import argparse
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig


device = "cuda:0"
max_seq_length = 512*4

ZEP_SYSTEM_PROMPT="""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

def formatting_prompts_func(example):
    output_texts = []
    conversations = example["conversations"]
    for convsation in conversations:
        q = convsation[0]['value']
        a = convsation[1]['value']
        text = f"[INST] {q} [/INST] {a}"
        output_texts.append(text)
    return output_texts

def formatting_zep_prompts_func(example):
    output_texts = []
    conversations = example["conversations"]
    for convsation in conversations:
        q = convsation[0]['value']
        a = convsation[1]['value']
        text = f"<s><|system|>\n{ZEP_SYSTEM_PROMPT}</s>\n<|user|>\n{q}</s>\n<|assistant|>\n{a}"
        #print(text)
        output_texts.append(text)
    return output_texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--check_steps", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--model_path", type=str, default="./llama2")
    parser.add_argument("--model", choices=['llama', 'zep',], default="llama")
    args = parser.parse_args()
    

    ALPACADATA = "./data/alpaca.json"
    GDATA = "./data/ganguli.json"
    df = [GDATA, ALPACADATA]

    model_path="./adapter/"+args.model+"_"+str(args.epochs)
    dataset = load_dataset("json", data_files=df, split="train").shuffle(seed=42)
    MODEL = args.model_path
    
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=MODEL,
                                                 device_map=device, torch_dtype=torch.bfloat16,
                                                 attn_implementation='flash_attention_2',
                                                 )
    
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=MODEL,
                                              use_fast=False)
    
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    
    batch_size = args.batch_size

    train_args = TrainingArguments(
        output_dir=model_path,
        num_train_epochs=args.epochs,              
        per_device_train_batch_size=batch_size, 
        per_device_eval_batch_size=4,
        save_steps=args.check_steps,
        gradient_accumulation_steps=16,
        learning_rate=2e-5,
        warmup_ratio=0.04,                
        weight_decay=0.,
        lr_scheduler_type="cosine",           
        logging_dir='./logs',            
        logging_steps=10,
        report_to="none"
    )
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    if args.model == "zep":
        response_template_string = "\n<|assistant|>"
        response_template = tokenizer.encode(response_template_string, add_special_tokens=False)[2:] 
    else:
        response_template = "[/INST]"
    
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    if args.model == 'zep':
        func = formatting_zep_prompts_func
    else:
        func = formatting_prompts_func
    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        formatting_func=func,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        dataset_kwargs={
            "add_special_tokens": False,
            "append_concat_token": False,
            },
        data_collator=collator,
        peft_config = peft_config
    )
    
    trainer.train(resume_from_checkpoint = False)
    trainer.save_model()



if __name__ == "__main__":
    main()
