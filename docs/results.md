# Results

All numbers are copied verbatim from the source fact sheet; each table carries its measurement condition. No number is reconciled, rounded, or invented.

## Decode throughput

Measured on the single-node deployment (Qwen3.6-35B-A3B-NVFP4, nvidia-official build, MTP-3 + marlin MoE backend + flashinfer + fp8 KV, flags per the stack table). Two recorded code-decode values are reported both ways because their median-vs-low labeling is not stated in the source.

| Workload | Value(s) | Condition |
|---|---|---|
| Code decode, MTP-3 recipe | **113.6 / 118.1 tok/s** | single node; two recorded values (median-vs-low labeling not stated in source); our measurements |
| Code decode, no-MTP baseline | **78.4 tok/s** | baseline against the two values above; improvement **+44.9% to +50.6%** (113.6 / 118.1 vs 78.4 tok/s, single stream) |
| Code decode (pre-deployment prediction) | **~103 tok/s** | a pre-deployment prediction from the community MTP-3 recipe; the 113.6 / 118.1 above are our measurements |
| Chinese prose decode | **83.7 / 93.3 / 87.5, mean 88.2 tok/s** | three runs (prose prompts); no no-MTP prose baseline was recorded. **Baseline mismatch:** the +12% note below is our note against the code baseline (78.4 tok/s), not a like-for-like prose comparison |

**Prediction vs measurement, stated openly:** the pre-deployment prediction from the community MTP-3 recipe was ~103 tok/s, while our deployment measurements were 113.6 / 118.1 tok/s. The prediction is a community-recipe forecast, not one of our measured values; we report both and do not reconcile them.

**Prose vs code (our note only):** the prose mean 88.2 tok/s sits about 12% above the 78.4 tok/s code baseline. Because no no-MTP prose baseline was recorded, this +12% is our note against the code baseline, marked "baseline mismatch", not a measured prose lift.

## MTP acceptance

Measured under the prose condition on the single-node deployment.

| Metric | Value |
|---|---|
| Mean acceptance length | **2.51** |
| Per-position acceptance (draft tokens 1/2/3) | **0.74 / 0.47 / 0.29** |

Acceptance length 2.51 = 1 + 0.74 + 0.47 + 0.29 rounded; the per-position rates are per-position acceptance rates on prose prompts.

## Memory / KV pool

Measured on the single-node deployment, fp8 KV, launch flags per the stack table.

| Item | Value | Condition |
|---|---|---|
| KV cache pool | **22.46 GiB = 1,054,918 tokens** | fp8 KV, flags per the stack table |
| Container host RAM | **7.2G** used against the hard cap | launch flag `--memory=55g` is a host-RAM hard cap in gigabytes; 7.2G here is the measured usage, a different quantity from the pool size (GiB) |
| GPU-memory utilization flag | **0.40** | launch flag |

## ChartQA quality gate

Measured with 40 questions, API path, seed 123, relaxed judging. The transformers 768×768-squash row is kept out of the 40-question comparison: its item count was not recorded, and the two evaluation conventions differ in serving stack and image preprocessing.

| Model variant | Score | Condition |
|---|---|---|
| Base NVFP4 @ vLLM | **87.5%** | 40 questions, seed 123, relaxed judging, API path |
| + chart-QA vision LoRA adapter (multi-LoRA) | **90.0%** | same gate; delta **+2.5**, a net one question (36/40 vs 35/40); no repeated measurement to confirm a stable lift. The adapter was loaded through vLLM multi-LoRA; its language-side modules applied, its vision-tower modules were ignored by vLLM at that version (logged as "no matching PunicaWrapper"), so the 90.0% reflects the language-side adapter only |
| Trap comparison: base @ transformers, 768×768 squash | **66.67%** | the 66.67% figure comes from a different item count that was not recorded, so it is kept out of the 40-question comparison. The two evaluation conventions differ in serving stack and image preprocessing (forced square resize vs aspect-ratio-preserving); we did not isolate the resize alone |

## What did not work

Negative results, copied verbatim.

| Attempt | Result |
|---|---|
| unsloth Fast NVFP4 quantization on this machine | **no increment over the nvidia-official NVFP4 build**. Community reports on this hardware: NVIDIA's official NVFP4 checkpoint 103.4 tok/s vs the unsloth NVFP4 checkpoint 87.6 tok/s (about −15%). This is unrelated to the 2.5× figure quoted for B200 — the marketing number does not transfer to GB10 |
| 27B dense model on one GB10 | a dense 27B NVFP4 model decodes at about **20 tok/s** single stream on this hardware in our runs; this is why the 35B-A3B MoE was locked in |
| MTP with `max_num_batched_tokens` left at 2048 | with MTP enabled the engine lowered `max_num_batched_tokens` to 2048, which is below the 2144-token attention block that the hybrid-mamba alignment needs, so the engine asserted at start-up; setting `--max-num-batched-tokens 4096` fixes it (root cause in `pitfalls.md`) |
| x86_64 vLLM image tag on the aarch64 host | submodule import failures via wrong Triton/PTXAS compile path |
| transformers-side 768×768-squash eval as a quality baseline | produced 66.67%; the item count for that run was not recorded, and the two evaluation conventions differ in serving stack and image preprocessing — we did not isolate the resize alone, so it is not a model-quality signal |
| Checkpoint-saving during GB10 training runs | epoch-boundary save spikes caused global OOM — three recorded deaths, all at epoch boundaries; `save_strategy=no` avoids the checkpoint-time memory spike (final weights are exported once at the end; no mid-run resume) |
| Serving inference through `PeftModel` (unmerged LoRA) on the base model | inference **20× slower** in our runs (4 min vs 12 s per item); for evaluation, run the merged model. multi-LoRA serving in vLLM is a separate path |
| `unsloth save_pretrained` as an archive | in our unsloth run, `save_pretrained` wrote only the config; the merged weights were in the merge output directory |
