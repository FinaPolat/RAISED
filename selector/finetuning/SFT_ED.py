import unsloth
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import torch
print(torch.cuda.is_available())  # Should return True if GPUs are available
print(torch.cuda.device_count())  # Number of GPUs available
import os
os.environ["HUGGING_FACE_HUB_TOKEN"] = "your_hugging_face_token_here"  # Replace with your actual token
os.environ.pop("TRANSFORMERS_OFFLINE", None)
from huggingface_hub import login
# Login directly with your token
login(token="your_hugging_face_token_here")
from torch.utils.tensorboard import SummaryWriter
import argparse
from unsloth import FastLanguageModel
import gc
gc.collect()
gc.collect()
torch.cuda.empty_cache()

tmpdir = os.environ.get("TMPDIR")
print(f"Temporary directory: {tmpdir}")
if tmpdir is None:
    print("TMPDIR environment variable is not set.")

hf_cache = os.environ.get("HF_HOME", "~/.cache/huggingface")
print(f"HF cache is at: {hf_cache}")


def create_message_column(row):
    """Create a conversational message column for the dataset."""
    messages = row["prompt"]
    messages.append(row["completion"][0])
    return {"messages": messages}


def format_dataset_chatml(row, tokenizer):
    # 'tokenizer.apply_chat_template' is a method that formats a list of chat messages into a single string.
    # 'add_generation_prompt' is set to False to not add a generation prompt at the end of the string.
    # 'tokenize' is set to False to return a string instead of a list of tokens.
    return {"text": tokenizer.apply_chat_template(row["messages"], add_generation_prompt=False, tokenize=False)}

def main():

    parser = argparse.ArgumentParser(description="Fine-tune LLM model for NER task")

    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to fine-tune the model")    
    parser.add_argument("--LLM_model", type=str, default="Qwen/Qwen3-8B", help="Name of the LLM model to fine-tune")
    parser.add_argument("--train_data", type=str, default="data/train.jsonl", help="Path to the training data")
    parser.add_argument("--val_data", type=str, default="/data/validation.jsonl", help="Path to the validation data")
    parser.add_argument("--ftuned_model_name", type=str, default="Qwen3-8B-ft", help="WandB project name")
    parser.add_argument("--logging_dir", type=str, default="logs", help="Directory to save logs")

    # Parse the arguments
    args = parser.parse_args()
    args_dict = vars(args)
    for arg, value in args_dict.items():
        print(f"{arg}: {value}")

    model_name = args.LLM_model

    data = load_dataset(
                        "json",
                        data_files={
                            "train": args.train_data,
                            "validation": args.val_data,
                        },
                        split={
                            "train": "train[:1000]",
                            "validation": "validation[:100]",
                        },
                    )

    print("Dataset loaded")
    print("Number of training samples:", len(data["train"]))
    print("Number of validation samples:", len(data["validation"]))
    print("Columns:", data.column_names)
    print("Sample:", data["train"][0])

    # Load the LLM model and tokenizer
    max_seq_length = 8000
    model, tokenizer = FastLanguageModel.from_pretrained(
                                                        model_name= model_name,
                                                        max_seq_length=max_seq_length,
                                                        load_in_4bit=True,
                                                        dtype=None,
                                                        )

    model = FastLanguageModel.get_peft_model(
                                        model,
                                        r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
                                        target_modules = ["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"],
                                        lora_alpha = 32,
                                        lora_dropout = 0.05, # Supports any, but = 0 is optimized
                                        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
                                        random_state = 3407,
                                        use_rslora = True, 
                                    )
    print("Model and tokenizer loaded", flush=True)

    # Prepare data for fine-tuning
    data = data.map(create_message_column)
    data = data.map(format_dataset_chatml, fn_kwargs={"tokenizer": tokenizer}, batched=True)
    data = data.remove_columns(["prompt", "completion", "messages"])
    print("Columns after mapping", data.column_names)
    print("Sample after mapping", data["train"][0])

    stf_config = SFTConfig(
                            per_device_train_batch_size = 1,
                            per_device_eval_batch_size = 1,
                            eval_accumulation_steps = 1,
                            gradient_accumulation_steps = 8,
                            max_grad_norm = 0.3,
                            warmup_steps = 50,
                            num_train_epochs = args.epochs, 
                            max_steps = -1,  # -1 means train for the number of epochs specified
                            learning_rate = 5e-6,
                            gradient_checkpointing = True,
                            fp16 = False,
                            bf16 = True,
                            save_strategy = "epoch",
                            logging_steps = 1,
                            optim = "paged_adamw_32bit",
                            weight_decay = 0.01,
                            lr_scheduler_type = "constant_with_warmup",
                            seed = 3407,
                            output_dir = f"{tmpdir}/temp_model_push",
                            report_to="none",  # Change to "wandb" to enable wandb logging
                        )
    log_dir = os.path.join(tmpdir, args.logging_dir)
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logging directory: {log_dir}", flush=True)
    writer = SummaryWriter(log_dir=log_dir)

    trainer = SFTTrainer(
            model = model,
            tokenizer = tokenizer,
            train_dataset=data["train"],
            eval_dataset=data["validation"],
            dataset_text_field = "text",
            max_seq_length = max_seq_length,
            dataset_num_proc = 2,
            packing = False, 
            args = stf_config,
                    )
    print("Starting training")    
    trainer.train()
    print("Training completed")
    # Log metrics to TensorBoard
    for log in trainer.state.log_history:
        step = log.get("step")
        if step is None:
            continue
        
        for key, value in log.items():
            if key == "step" or value is None:
                continue
            # Replace "/" in key with "_" if you want
            tensorboard_key = key.replace("/", "_")
            writer.add_scalar(tensorboard_key, value, step)

    print("Evaluating the model")
    metrics =trainer.evaluate()  
    print("Evaluation completed", flush=True)
    step = trainer.state.global_step
    for key, value in metrics.items():
        if value is not None:
            writer.add_scalar(f"eval/{key}", value, step)

    writer.close()

    model.push_to_hub_merged(
                    f"your_HF_username/{args.ftuned_model_name}",
                    tokenizer,
                    save_method="merged_16bit"
                )
    
    print("Fine-tuning process completed successfully.", flush=True)
    # print("Starting second push to hub for GGUF format.", flush=True)
    # model.push_to_hub_gguf(
    #                 f"your_HF_username/{args.ftuned_model_name}-gguf", 
    #                 tokenizer, 
    #                 quantization_method = "q4_k_m")

    print("Pushed to the hub")


if __name__ == "__main__":
    main()