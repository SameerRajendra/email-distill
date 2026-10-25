"""LoRA fine-tune Gemma 4 E4B on the email rows with DDP or FSDP2, under torchrun. Launched by train_dist_modal.py.

Same task and hyper-parameters as train_modal.py, but plain TRL + PEFT in bf16 (no 4-bit, no Unsloth) so the
trainer can be sharded. Scaling mode times `--timed` steps after `--warm` warm-up steps, then profiles 2 more
steps; rank 0 appends one JSON line. Per-GPU batch is fixed, so adding GPUs grows the global batch (weak scaling).
"""
import argparse
import json
import os
import time

import torch
import torch.distributed as dist
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer

MODEL, SEED, BATCH = "unsloth/gemma-4-E4B-it", 3407, 2
# Language-model projections only (the vision and audio towers have q_proj/k_proj too): same set as train_modal.py.
LORA_TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


class Bench(TrainerCallback):
    """Throughput between steps `warm` and `warm + timed`; NCCL share of CUDA kernel time over the 2 steps after."""

    def __init__(self, warm, timed):
        self.warm, self.timed, self.m, self.marks = warm, timed, {}, []

    def on_train_begin(self, args, state, control, **kw):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()    # after sharding, so the one-off full-model load is not counted

    def on_log(self, args, state, control, logs=None, **kw):   # logged every step: each log is a synced timestamp
        if state.global_step in (self.warm, self.warm + self.timed):
            self.marks.append((time.perf_counter(), logs["num_tokens"]))   # TRL's global real-token count

    def on_step_begin(self, args, state, control, **kw):
        if state.global_step == self.warm + self.timed:
            self.prof = torch.profiler.profile()   # CPU and CUDA activity, whichever exist
            self.prof.start()

    def on_step_end(self, args, state, control, **kw):
        if state.global_step == self.warm + self.timed + 2:
            self.prof.stop()
            kernels = [e for e in self.prof.key_averages() if e.device_type == torch.autograd.DeviceType.CUDA]
            total = sum(e.self_device_time_total for e in kernels)
            nccl = sum(e.self_device_time_total for e in kernels if "nccl" in e.key.lower())
            self.m["comm_share"] = nccl / total if total else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=["ddp", "fsdp"], required=True)
    ap.add_argument("--data", required=True, help="JSON {train: [...], test: [...]} of chat rows")
    ap.add_argument("--out", required=True, help="JSONL file rank 0 appends its record to")
    ap.add_argument("--final", metavar="ADAPTER_DIR", help="train 2 epochs with eval and save the adapter here")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--warm", type=int, default=10)
    ap.add_argument("--timed", type=int, default=20)
    a = ap.parse_args()
    rank, world, cuda = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), torch.cuda.is_available()
    fsdp = a.strategy == "fsdp"

    rows = json.load(open(a.data, encoding="utf-8"))
    def dataset(part):
        return Dataset.from_list([{"prompt": r["messages"][:-1], "completion": r["messages"][-1:]} for r in part])

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map={"": int(os.environ["LOCAL_RANK"])} if cuda else None)
    bench = None if a.final else Bench(a.warm, a.timed)

    # prompt/completion rows: TRL renders the chat template and, by default, takes the loss on the completion only.
    trainer = SFTTrainer(
        model=model, processing_class=tok, train_dataset=dataset(rows["train"]),
        eval_dataset=dataset(rows["test"]) if a.final else None, callbacks=[bench] if bench else None,
        peft_config=LoraConfig(r=16, lora_alpha=16, lora_dropout=0, bias="none", task_type="CAUSAL_LM",
                               target_modules=LORA_TARGETS),
        args=SFTConfig(
            output_dir="/tmp/trainer", use_cpu=not cuda, bf16=True, seed=SEED, report_to="none", save_strategy="no",
            max_length=2048, per_device_train_batch_size=BATCH, per_device_eval_batch_size=BATCH, learning_rate=2e-4,
            lr_scheduler_type="linear", warmup_steps=5, weight_decay=0.001, optim="adamw_torch",
            num_train_epochs=2, max_steps=-1 if a.final else a.warm + a.timed + 2,
            logging_steps=5 if a.final else 1, eval_strategy="epoch" if a.final else "no",
            # FSDP2 checkpoints per decoder layer itself (transformers advises against gradient_checkpointing there).
            gradient_checkpointing=not fsdp, fsdp=True if fsdp else None,
            fsdp_config={"version": 2, "auto_wrap_policy": "TRANSFORMER_BASED_WRAP", "activation_checkpointing": True,
                         "transformer_layer_cls_to_wrap": "Gemma4TextDecoderLayer"} if fsdp else None))
    if rank == 0:
        row = trainer.train_dataset[0]
        print("trained span:", tok.decode([t for t, m in zip(row["input_ids"], row["labels"]) if m != -100]))
    trainer.train()
    if a.final:
        trainer.save_model(a.final)

    peak = None
    if cuda:
        peak = torch.tensor([torch.cuda.max_memory_allocated() / 2**30], device="cuda")
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        peak = peak.item()
    if rank == 0:
        import accelerate, peft, transformers, trl
        rec = {"strategy": a.strategy, "world_size": world, "model": a.model, "per_gpu_batch": BATCH,
               "gpu": torch.cuda.get_device_name(0) if cuda else "cpu", "peak_gib": peak,
               "final_loss": next(h["loss"] for h in reversed(trainer.state.log_history) if "loss" in h),
               "versions": {m.__name__: m.__version__ for m in (torch, transformers, trl, peft, accelerate)}}
        if bench:
            (t0, n0), (t1, n1) = bench.marks
            sps = a.timed * BATCH * world / (t1 - t0)
            rec.update(step_time_s=(t1 - t0) / a.timed, samples_per_s=sps, tokens_per_s=(n1 - n0) / (t1 - t0),
                       tokens_per_s_per_gpu=(n1 - n0) / (t1 - t0) / world, comm_share=bench.m["comm_share"])
        else:
            rec["log_history"] = [h for h in trainer.state.log_history if "loss" in h or "eval_loss" in h]
        with open(a.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
