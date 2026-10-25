# email-distill

**Can a 4B model on a laptop GPU replace a frontier-LLM API for a narrow email-extraction job?**

## The problem

Companies pipe every inbound email or support ticket through a frontier LLM API. That costs money per
call, adds latency, and sends customer data outside the company. For a narrow, well-defined extraction
task the large model may be overkill.

This repo tests that on a concrete case: my job-application tracker reads application mail and returns,
per email, `kind` (applied / rejected / interview / incomplete / other), `company`, `position` and
`req_id`. Claude is the **teacher**. Its labels are distilled into **Gemma 4 E4B** with QLoRA, exported
to GGUF and served with llama.cpp on a consumer GPU. The questions:

- How closely does the model agree with the teacher, before and after fine-tuning?
- How often does it produce valid JSON on its own (no grammar constraint)?
- What does Q4_K_M vs Q8_0 quantization cost in agreement?
- What are latency and tokens/s, and what would the same traffic cost through an API?

The prompt, output schema and email rendering are the tracker's own (`schema.py`), so results transfer to
it. Claude's labels are a reference, not ground truth: agreement with the teacher is what is measured.

## Pipeline

| Step | Script | Where | Output |
|---|---|---|---|
| 1 | `export.py` | PC (Thunderbird add-on on 127.0.0.1) | `data/raw.jsonl` |
| 2 | `redact.py` | PC | `data/redacted.jsonl` |
| 3 | `label.py` | PC, calls Claude via the `claude` CLI | `data/labels.jsonl` |
| 4 | `prepare.py` | PC | `data/train.jsonl`, `data/test.jsonl` (oldest 80% / newest 20%) |
| 5 | `train_modal.py` | Modal, L4 GPU | LoRA adapter + GGUF Q4_K_M and Q8_0 in a Modal Volume |
| 6 | `eval.py`, or `eval_modal.py` | PC GPU, or Modal L4; llama.cpp | `results/<tag>.json` |
| 7 | `report.py` | PC | `results/REPORT.md` |

`schema.py` holds the prompt, schema and `format_email` that every step shares.

## Privacy design: what leaves the PC, and when

| Data | Leaves the PC? |
|---|---|
| Raw emails (`data/raw.jsonl`) | Never. Fetched from Thunderbird over localhost. |
| Redacted emails | Yes, twice: to Claude for labelling (step 3, through your own `claude` login), and, as the train/test rows, to Modal for training (step 5). |
| Modal training rows | Sent as an in-memory function argument. This code never writes them to a Modal Volume; only the adapter and GGUFs are persisted. Modal handles the call while it runs; read its data-retention policy if that matters to you. |
| Evaluation | `eval.py`: local only. `eval_modal.py`: the test rows go to Modal as an in-memory argument, like the training rows; no Volume holds them. |
| This repo | Contains no email data (`data/`, models and result JSON are git-ignored). |

What `redact.py` removes before anything is sent: your own names, addresses and phone numbers (from the
git-ignored `redact.local.json`), every URL beyond scheme+host, every email address beyond `@domain`, US-style
phone numbers, and digit runs of 6+. It prints how often each rule fired. Known limits: it is regex-based, so
recruiters' names and company names stay, and purely numeric requisition ids become `[NUM]`
(alphanumeric ones such as `JR2026249` survive). Skim `data/redacted.jsonl` before step 3.

## Run it (Windows)

```
python -m pip install -r requirements.txt
modal setup

python export.py --days 365
copy redact.example.json redact.local.json
notepad redact.local.json
python redact.py
python label.py
python prepare.py

modal run train_modal.py
```

Thunderbird must be running for `export.py`. Put your own names, email addresses and phone numbers in
`redact.local.json`. Re-run `label.py` if it reports skipped batches. `train_modal.py` ends by printing
the exact `modal volume get` commands.

Fetch the training outputs (the commands `train_modal.py` prints):

```
modal volume get email-distill-models email-distill-Q4_K_M.gguf models\
modal volume get email-distill-models email-distill-Q8_0.gguf models\
modal volume get email-distill-models adapter models\adapter
```

Evaluate the stock model and the tuned one under the same conditions. The stock GGUFs (for example from
`unsloth/gemma-4-E4B-it-GGUF`) go in `models\`; their quantization recipe is not ours, so base-vs-tuned also
includes any recipe difference.

```
python eval.py ^
  --model models\gemma-4-E4B-it-Q4_K_M.gguf --tag base-q4_k_m ^
  --model models\gemma-4-E4B-it-Q8_0.gguf   --tag base-q8_0 ^
  --model models\email-distill-Q4_K_M.gguf  --tag tuned-q4_k_m ^
  --model models\email-distill-Q8_0.gguf    --tag tuned-q8_0
python report.py --in-usd-per-mtok <price> --out-usd-per-mtok <price>
```

`eval.py` starts `llama-server` itself (`--llama-server`, default `D:\LLM models\llama.cpp\llama-server.exe`,
flags `-ngl 99 -c 4096 -fa on --jinja -np 1`, port 8090). `--url http://127.0.0.1:8080 --tag name` evaluates
a server you already started instead.

### Evaluate on Modal

On a slow home link, downloading the 5-8 GB tuned GGUFs just to score them is impractical, so
`eval_modal.py` runs the same measurement next to the models. Per tag it starts the official prebuilt
`ghcr.io/ggml-org/llama.cpp:server-cuda` image on an L4 and calls `eval.py`'s `serve` and `evaluate`
unchanged. The tuned GGUFs are read from the `email-distill-models` Volume. The stock model
(`gemma-4-E4B-it-Q4_K_M.gguf` from `lmstudio-community/gemma-4-E4B-it-GGUF`, the file used locally) is
downloaded into a second Volume, `email-distill-hf-cache`, on the first run only. `data/test.jsonl` is read
on the PC and sent as an argument. Only the results come back: `results/<tag>.json`, with an extra
`hardware` field, so `report.py` works as before. The three tags run in parallel, one L4 each. Latency is
still timed one request at a time; only the JSON-validity pass uses 4 parallel server slots.

```
modal run eval_modal.py --limit 20     # speed check, nothing saved
modal run eval_modal.py
modal run eval_modal.py --tags tuned-q4_k_m
```

The default is `base-q4_k_m,tuned-q4_k_m,tuned-q8_0`. Accuracy and valid-JSON rate are
meant to be comparable with `eval.py` runs (same model file, same prompts, temperature 0). Latency and
tokens/s from Modal are L4 numbers, not laptop numbers: do not mix them with results from `eval.py` in one
table.

## What is measured

All metrics compare the model with the teacher's labels on the held-out newest 20% of mail.

- **Accuracy, macro-F1, per-class precision/recall, confusion matrix**: `kind`, from the schema-constrained
  pass (how the tracker runs it). Macro-F1 averages the classes that occur in the test set.
- **Employer match**: for emails the teacher did not label `other`; normalised exact match or rapidfuzz
  `token_set_ratio >= 90`.
- **Valid JSON (unconstrained)**: a second pass without `response_format`; valid means it parses and has the
  schema's keys with `kind` in the enum.
- **Latency p50/p95** per email, **decode and prefill tokens/s** (median of llama-server's `timings`),
  **total prompt/output tokens**. llama-server caches the shared system prompt, so prefill tok/s covers the
  uncached part of each request.
- **Cost per 1k emails**: measured tokens per email times the API prices you pass to `report.py`. Tokens are
  counted by Gemma's tokenizer, so this approximates an API bill. The local model has no per-call cost.

## Distributed training

**The problem.** Multi-GPU fine-tuning cost depends on how well training scales, and that depends on the
parallelism strategy (DDP replicates the model and all-reduces gradients; FSDP shards weights, gradients and
optimizer state and gathers each layer when it is needed) and on the interconnect (PCIe-only L40S vs NVLink
H100). The workload is this repo's own job: LoRA-tuning Gemma 4 E4B on the labelled emails.

**Design.** `train_dist.py` is plain TRL + PEFT in bf16 (no 4-bit, no Unsloth), launched by `torchrun` with
`--strategy ddp|fsdp` (FSDP2 through the Trainer's `fsdp_config`, one shard group per decoder layer). Task, data,
LoRA r=16/alpha=16, learning rate 2e-4, 2 epochs and seed match `train_modal.py`; the optimizer is plain AdamW
instead of 8-bit AdamW, and the loss is on the assistant's JSON only (TRL prompt-completion rows).
`train_dist_modal.py` starts one Modal container per GPU type and count (`L40S:N`, `H100:N`, N = 1, 2, 4), runs
both strategies in it and records, per run: samples/s and tokens/s (total and per GPU), step time, peak memory
per GPU (max over ranks), final train loss, the share of CUDA kernel time spent in NCCL (`torch.profiler`, 2
steps, rank 0). (`nvidia-smi topo -m` does not run inside Modal's sandbox, so the link type is not recorded.) The per-GPU batch is fixed (weak scaling); a scaling run trains 32
steps, not 2 epochs. `--final` trains the 2-epoch model on 4 GPUs with the fastest measured strategy, then merges
it on a CPU container and exports the same `email-distill-Q4_K_M.gguf` / `-Q8_0.gguf` into the Volume, so
`eval_modal.py` works unchanged. Both modes print a cost estimate from Modal's list prices before launching.

```
modal run train_dist_modal.py --gpus l40s --counts 1,2      # cheap smoke test first
modal run train_dist_modal.py                               # the grid: l40s,h100 x 1,2,4 -> results/scaling/<gpu>.jsonl
python report_scaling.py                                    # -> results/scaling/SCALING.md and scaling_efficiency.png
modal run train_dist_modal.py --final --final-gpu h100      # add --strategy ddp|fsdp to override the pick
```

**Results** (`results/scaling/SCALING.md`). Tokens/s per GPU, efficiency vs 1 GPU in brackets:

| | 1 GPU | 2 GPUs | 4 GPUs |
|---|---|---|---|
| L40S DDP | 2,246 | 2,053 (91%) | 1,910 (85%) |
| L40S FSDP2 | 1,757 | 869 (49%) | 815 (46%) |
| H100 DDP | 2,734 | 1,610 (59%, single run, likely a bad node) | 2,383 (87%) |
| H100 FSDP2 | 2,183 | 1,960 (90%) | 1,970 (90%) |

DDP wins for LoRA: only the adapter gradients are all-reduced. FSDP2 all-gathers the frozen base for every
layer; NCCL is about 70% of its kernel time on both GPUs, but on H100 it overlaps compute and over PCIe it
cannot. FSDP2 peaks at 30 GiB on 1 GPU (about 8B parameters in fp32), so the frozen base was probably upcast;
that inflates its traffic but does not change the winner. The final model was trained on 4x L40S with DDP.

**Scope.** One node, at most 4 GPUs. LoRA gradients are tiny (adapter only), which bounds the communication share
for DDP; FSDP still gathers the frozen base weights every layer, and a full fine-tune would stress the
interconnect far more. Rows have different lengths, so on more GPUs every step waits for the longest batch: that
loss is counted in "efficiency" next to communication, which is why the NCCL share is reported separately. The
study measures throughput, not quality; as in `train_modal.py`, the rows reach the container as a function
argument, and `torchrun`'s workers read them from the container's own `/tmp`, never a Volume.

## Results

342 held-out emails (newest 20%), llama.cpp on an L4; full tables and confusion matrices in `results/REPORT.md`.

| Model | Agreement with teacher | Macro-F1 | Valid JSON, unconstrained | p50 latency |
|---|---|---|---|---|
| Stock Gemma 4 E4B Q4_K_M | 98.0% | 0.90 | 6.1% | 4.2 s |
| Tuned Q4_K_M | 96.8% | 0.74 | 99.7% | 4.3 s |
| Tuned Q8_0 | 97.7% | 0.76 | 99.7% | 5.5 s |

With llama.cpp's json_schema constraint the stock model is already as accurate as the tuned one; the
fine-tune's gain is format adherence where no grammar can be enforced. Macro-F1 is dominated by the interview
class, which has 2 test emails. The tuned models give unparseable output on 8 of 342 emails even under the
constraint, most likely by looping to the 512-token cap; raw outputs are not saved yet, so that is unconfirmed.
