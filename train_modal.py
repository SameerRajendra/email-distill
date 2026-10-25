"""QLoRA-tune Gemma 4 E4B on the teacher-labelled emails on a Modal L4 GPU, then export GGUF.

Privacy: the rows travel to the container as a function argument (memory only) and are never written to
a Volume. Only the LoRA adapter and the two GGUFs are persisted, in the Volume `email-distill-models`.

    modal run train_modal.py
"""
import json
import subprocess
import sys
from pathlib import Path

import modal

MODEL = "unsloth/gemma-4-E4B-it"
VOLUME = "email-distill-models"
LLAMA_CPP = "/opt/llama.cpp"
MAX_SEQ, SEED = 2048, 3407

# The transformers / tokenizers / huggingface_hub pins and timm follow Unsloth's Gemma 4 notebook. llama.cpp
# is built (CPU only, quantizer only) because that notebook lists just Q8_0/BF16/F16 for Gemma 4 GGUF export,
# not Q4_K_M: we convert to BF16 ourselves and quantize with llama-quantize.
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("git", "build-essential", "cmake")
         # Exact Unsloth pair, straight from pypi.org: an unpinned install resolved an older unsloth_zoo that
         # imports trl's since-removed ConstantLengthDataset and fails at startup.
         .pip_install("unsloth==2026.10.8", "unsloth_zoo==2026.10.8", "datasets", index_url="https://pypi.org/simple")
         .pip_install("transformers==5.5.0", "tokenizers>=0.22.0,<=0.23.0", "timm", extra_options="--no-deps")
         .pip_install("huggingface_hub>=1.5.0,<2.0")
         .run_commands(f"git clone --depth 1 https://github.com/ggml-org/llama.cpp {LLAMA_CPP}",
                       f"cmake -S {LLAMA_CPP} -B {LLAMA_CPP}/build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF",
                       f"cmake --build {LLAMA_CPP}/build --target llama-quantize -j"))
app = modal.App("email-distill", image=image)
volume = modal.Volume.from_name(VOLUME, create_if_missing=True)


@app.function(gpu="L4", timeout=2 * 60 * 60, memory=32768, volumes={"/out": volume})
def train(train_rows: list[dict], test_rows: list[dict]) -> list[dict]:
    """Fine-tune, save the adapter, export GGUFs. Returns the training log (loss per logging step)."""
    from unsloth import FastModel                   # must be imported before trl / transformers
    from unsloth.chat_templates import get_chat_template, train_on_responses_only
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = FastModel.from_pretrained(MODEL, max_seq_length=MAX_SEQ, load_in_4bit=True,
                                                 full_finetuning=False)
    model = FastModel.get_peft_model(model, finetune_vision_layers=False, finetune_language_layers=True,
                                     finetune_attention_modules=True, finetune_mlp_modules=True,
                                     r=16, lora_alpha=16, lora_dropout=0, bias="none", random_state=SEED)
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")     # the non-thinking template

    def to_dataset(rows):
        # <bos> is stripped because the trainer's tokenizer adds it again.
        return Dataset.from_list([{"text": tokenizer.apply_chat_template(r["messages"], tokenize=False)
                                   .removeprefix("<bos>")} for r in rows])

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=to_dataset(train_rows), eval_dataset=to_dataset(test_rows),
        args=SFTConfig(dataset_text_field="text", num_train_epochs=2, learning_rate=2e-4,
                       per_device_train_batch_size=2, per_device_eval_batch_size=2, gradient_accumulation_steps=4,
                       warmup_steps=5, optim="adamw_8bit", weight_decay=0.001, lr_scheduler_type="linear",
                       logging_steps=5, eval_strategy="epoch", seed=SEED, report_to="none", output_dir="/tmp/trainer"))
    trainer = train_on_responses_only(trainer, instruction_part="<|turn>user\n", response_part="<|turn>model\n")
    row = trainer.train_dataset[0]      # only the assistant's JSON should be left unmasked
    print("trained span:", tokenizer.decode([t for t, label in zip(row["input_ids"], row["labels"]) if label != -100]))
    trainer.train()

    model.save_pretrained("/out/adapter")
    tokenizer.save_pretrained("/out/adapter")
    model.save_pretrained_merged("/tmp/merged", tokenizer, save_method="merged_16bit")
    subprocess.run([sys.executable, f"{LLAMA_CPP}/convert_hf_to_gguf.py", "/tmp/merged",
                    "--outfile", "/tmp/bf16.gguf", "--outtype", "bf16"], check=True)
    for quant in ("Q4_K_M", "Q8_0"):
        subprocess.run([f"{LLAMA_CPP}/build/bin/llama-quantize", "/tmp/bf16.gguf",
                        f"/out/email-distill-{quant}.gguf", quant], check=True)
    volume.commit()
    return [h for h in trainer.state.log_history if "loss" in h or "eval_loss" in h]


@app.local_entrypoint()
def main():
    def rows(name):
        return [json.loads(line) for line in (Path(__file__).parent / "data" / name).read_text(encoding="utf-8").splitlines()]

    for h in train.remote(rows("train.jsonl"), rows("test.jsonl")):
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in h.items() if k in ("epoch", "step", "loss", "eval_loss")})
    print("\nFetch the results (run in this folder):")
    print(f"  modal volume get {VOLUME} email-distill-Q4_K_M.gguf models\\")
    print(f"  modal volume get {VOLUME} email-distill-Q8_0.gguf models\\")
    print(f"  modal volume get {VOLUME} adapter models\\adapter")
