# python train.py --config fsdp.yaml
import logging
from dataclasses import dataclass, field
import os
import random
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments
from trl.cli import  TrlParser # 커맨드라인 인자 파싱 유틸

from transformers import(
    AutoModelForCausalLM,
    BitsAndBytesConfig, # 양자화 설정
    set_seed, # 재현성 고정
)
from trl import setup_chat_format # 대화 포맷 템플릿 세팅 
from peft import LoraConfig

from trl import(
    SFTTrainer # Supervised Fine-Tuning 전용
)

@dataclass
class ScriptArguments:
    # 필수 값들
    model_id: str = field(
        default="Bllossom/llama-3-Korean-Bllossom-70B",
        metadata={"help": "base model (HF Hub id or local path)"}
    )
    dataset_path: str = field(
        default=".",
        metadata={"help": "folder that has train_dataset.json / test_dataset.json"}
    )
    max_seq_length: int = field(
        default=4096,
        metadata={"help": "max input length after chat-template is applied"}
    )

# Anthropic/Vicuna 스타일 대화 템플릿
# system, user, assistant 메시지를 순회하며 프롬프트 형식으로 변경
LLAMA_3_CHAT_TEMPLATE = (
    "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
            "{{ message['content'] }}" 
        "{% elif message['role'] == 'user' %}"
            "{{ '\n\nHuman: ' + message['content'] +  eos_token }}"
        "{% elif message['role'] == 'assistant' %}"
            "{{ '\n\nAssistant: '  + message['content'] +  eos_token  }}"
        "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '\n\nAssistant: ' }}"
    "{% endif %}"
)



def training_function(script_args, training_args):
    
    # Dataset
    # >>>>>>>>>>>>>>>>>>>
    
    train_dataset = load_dataset(
        "json",
        data_files = os.path.join(script_args.dataset_path, "train_dataset.json"),
        split="train"
    )
    
    test_dataset = load_dataset(
        "json",
        data_files = os.path.join(script_args.dataset_path, "test_dataset.json"),
        split="train"
    )
    
    # Tokenizer
    # >>>>>>>>>>>>>>>>>>>
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_id, use_fast = True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = LLAMA_3_CHAT_TEMPLATE
    
    # template dataset -> json 형태의 messages를 단일 문자열로 포맷팅
    def template_dataset(examples):
        return{"text": tokenizer.apply_chat_template(examples["messages"], tokenizer=False)}
    
    train_dataset = train_dataset.map(template_dataset, remove_columns=["messages"])
    test_dataset = test_dataset.map(template_dataset, remove_columns=["messages"])
    
    # print random sample
    with training_args.main_process_first(desc="Log a few random samples from the processed training set"):
        for index in random.sample(range(len(train_dataset)), 2):
            print(train_dataset[index]["text"])

    # Model
    # >>>>>>>>>>>>>>>>>>>    
    
    # 양자화 형식
    torch_dtype = torch.bfloat16
    quant_storage_dtype = torch.bfloat16
    
    # 4bit 압축된 파라미ㅓ를 bfloat16으로 복원하여 연산
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_quant_storage=quant_storage_dtype,
    )

    # 모델 로더
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_id,
        quantization_config=quantization_config, 
        attn_implementation="sdpa", # 어텐션 연산 설정 (sdpa, flash_attention_2)
        torch_dtype=quant_storage_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,  # this is needed for gradient checkpointing
    )
    
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # PEFT
    # >>>>>>>>>>>>>>>>>>>    
    # LoRA config based on QLoRA paper & Sebastian Raschka experiment
    peft_config = LoraConfig(
        lora_alpha=8,
        lora_dropout=0.05,
        r=16,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        # modules_to_save = ["lm_head", "embed_tokens"] # add if you want to use the Llama 3 instruct template
    )


    # Training
    # >>>>>>>>>>>>>>>>>>>   
    trainer = SFTTrainer(
        model = model,
        args = training_args,
        train_dataset = train_dataset,
        dataset_text_field = "text",
        eval_dataset = test_dataset,
        peft_config=peft_config,
        max_seq_length=script_args.max_seq_length,
        tokenizer = tokenizer,
        packing = True,
        dataset_kwargs = {
            "add_special_tokens": False,  # We template with special tokens
            "append_concat_token": False,  # No need to add additional separator token
        }
    )
    
    if trainer.accelerator.is_main_process:
        trainer.model.print_trainable_parameters()
        
    trainer.save_model()

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_and_config()    
    
    # set use reentrant to False
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}
    # set seed
    set_seed(training_args.seed)

    # launch training
    training_function(script_args, training_args)