import os, json
import torch
from datasets import Dataset
import pandas
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

def load_data(path: str):
    import deft
    with open(path) as fp:
        corpus = json.loads(fp.read())
    template = deft.lm_templates[0]
    corpus = [{"text": deft.get_prompt(template, instance, include_correct_answers=True, include_full_answers=True)} for instance in corpus]
    dataset = Dataset.from_pandas(pandas.DataFrame(data=corpus))
    return dataset

def finetune_lora(model_name: str, 
        train_dataset_name: str, 
        eval_dataset_name: str, 
        new_model: str, 
        run_name: str,
        lora_r:int = 4, 
        lora_alpha: int = 16, 
        lora_dropout: float = 0.05, 
        use_4bit: bool = True, 
        bnb_4bit_compute_dtype: str = "float16", 
        bnb_4bit_quant_type: str = "nf4", 
        use_nested_quant: bool = False, 
        output_dir: str = "./results", 
        num_train_epochs: int = 1, 
        fp16: bool = False, 
        bf16: bool = True, 
        batch_size: int = 4,
        micro_batch_size: int = 4,
        gradient_checkpointing: bool = True, 
        max_grad_norm: float = 0.3, 
        learning_rate: float = 3e-4, 
        weight_decay: float = 0.001, 
        optim: str = "paged_adamw_32bit", 
        lr_scheduler_type: str = "cosine", 
        max_steps: int = -1, 
        warmup_ratio: float = 0.05, 
        group_by_length: bool = True, 
        save_steps: int = 0, 
        logging_steps: int = 1, 
        max_seq_length: int = 256, 
        packing: bool = False, 
        device_map: str = '{"":0}',
        train_on_completions_only: bool = False):

    device_map = json.loads(device_map)
    train_dataset = load_data(train_dataset_name)
    #eval_dataset = load_data(eval_dataset_name)

    # Load tokenizer and model with QLoRA configuration
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

    # Check GPU compatibility with bfloat16
    if compute_dtype == torch.float16 and use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

    # Load LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "v_proj"],
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Set training parameters
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=batch_size // micro_batch_size,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        run_name=run_name,
        report_to="wandb",
    )

    if train_on_completions_only:
        response_template = "Réponse(s) :"
        tokens =  tokenizer.tokenize(response_template, add_special_tokens=False)
        token_ids = tokenizer.encode(response_template, add_special_tokens=False)
        print(list(zip(tokens, token_ids)))
        collator = DataCollatorForCompletionOnlyLM(token_ids[1:], tokenizer=tokenizer)
    else:
        collator = None

    # Set supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        #eval_dataset=eval_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=packing,
        data_collator=collator,
    )

    # Train model
    trainer.train()

    # Save trained model
    trainer.model.save_pretrained(new_model)

def load_for_inference(model_name: str, 
        lora_model: str, 
        use_4bit: bool = True, 
        bnb_4bit_compute_dtype: str = "float16", 
        bnb_4bit_quant_type: str = "nf4", 
        use_nested_quant: bool = False, 
        device_map='{"":0}'):

    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        #low_cpu_mem_usage=True,
        quantization_config=bnb_config,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map=json.loads(device_map),
    )
    model = PeftModel.from_pretrained(base_model, lora_model)
    #model = model.merge_and_unload()

    # Reload tokenizer to save it
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer

if __name__ == '__main__':
    import fire
    #print(load_data("../json/train.json")['text'][0])
    fire.Fire(finetune_lora)
