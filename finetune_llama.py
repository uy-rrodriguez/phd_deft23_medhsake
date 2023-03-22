# adapted from https://github.com/tloen/alpaca-lora/
import os
import uuid
import json

from datasets import load_dataset
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model

from deft import linearize_instance

# reasonable batch size with 80GB GPU
# 7B -> 24
# 13B -> 12
# 30B -> 6
# 65B -> 1

def main(LLAMA_VARIANT : int = 65,
    MICRO_BATCH_SIZE : int = 1,
    BATCH_SIZE : int = 24,
    EPOCHS : int = 1,
    LEARNING_RATE : float = 3e-4,
    CUTOFF_LEN : int = 256,
    WARMUP_RATIO : float = 0.05,
    IS_INT8 : bool = True,
    LORA_R : int = 4,
    LORA_ALPHA : int = 16,
    LORA_DROPOUT : int = 0.05,
    valid_steps : int = 20,
    PROMPT : str = '''Ceci est une question de QCM de l\'examen de pharmacie. Réponds avec la ou les lettres correspondant à la bonne réponse.\n\n%s''',
    output_dir : str = 'deft_models',
    train_json : str = '../../json/train.json',
    dev_json : str = '../../json/dev.json',
    ):
#PROMPT = '''Ceci est une question de QCM de l\'examen de pharmacie. Réponds avec la ou les lettres correspondant à la bonne réponse.\n\n## Question 1\n\n%s'''

    GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE


    BACKBONE = "decapoda-research/llama-%db-hf" % LLAMA_VARIANT
    OUTPUT_PATH = "%s/deft_%s_lora_%s" % (output_dir, BACKBONE.split('/')[-1], str(uuid.uuid4()))
    print(OUTPUT_PATH)
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    model = LlamaForCausalLM.from_pretrained(
        BACKBONE,
        load_in_8bit=IS_INT8,
        device_map="auto",
    )


    tokenizer = LlamaTokenizer.from_pretrained(BACKBONE, add_eos_token=True)

    if IS_INT8:
        model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    data = load_dataset("json", data_files="../../json/train.json")
    data_valid = load_dataset("json", data_files="../../json/dev.json")
    data['validation'] = data_valid['train']

    def generate_prompt(data_point):
        return PROMPT % linearize_instance(data_point, include_correct_answers=True)

    print(generate_prompt(data['train'][0]))

    data = data.shuffle().map(
        lambda data_point: tokenizer(
            generate_prompt(data_point),
            truncation=True,
            max_length=CUTOFF_LEN,
            padding="max_length",
        )
    )

    trainer = transformers.Trainer(
        model=model,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        args=transformers.TrainingArguments(
            do_eval=True,
            evaluation_strategy='steps',
            eval_steps=valid_steps,
            per_device_train_batch_size=MICRO_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            warmup_ratio=WARMUP_RATIO,
            num_train_epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            fp16=True,
            logging_steps=1,
            output_dir=OUTPUT_PATH,
            save_total_limit=3,
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False
    trainer.train(resume_from_checkpoint=False)

    model.save_pretrained(OUTPUT_PATH)

    with open("%s/training_config.json" % OUTPUT_PATH, 'w') as fp:
        fp.write(json.dumps({
            'backbone': BACKBONE,
            'micro_batch_size': MICRO_BATCH_SIZE,
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'learning_rate': LEARNING_RATE,
            'cutoff_len': CUTOFF_LEN,
            'warmup_ratio': WARMUP_RATIO,
            'llama_variant': LLAMA_VARIANT,
            'is_int8': IS_INT8,
            'lora_r': LORA_R,
            'lora_alpha': LORA_ALPHA,
            'lora_dropout': LORA_DROPOUT,
            'prompt': PROMPT,
            'output_path': OUTPUT_PATH
        }, indent=4))

    print(OUTPUT_PATH)

if __name__ == '__main__':
    import fire
    fire.Fire(main)
