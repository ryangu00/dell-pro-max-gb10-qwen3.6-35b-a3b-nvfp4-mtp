# Pitfalls

Each pitfall is recorded as **symptom → root cause → fix → how we found it**. Values are copied verbatim from the source; where the source does not state how a pitfall was found, the "how we found it" line says so explicitly.

---

## 1. AssertionError at MTP launch

- **Symptom:** vLLM raises an `AssertionError` at startup when MTP-3 speculative decoding is enabled.
- **Root cause:** With MTP enabled the engine lowered `max_num_batched_tokens` to 2048, which is below the 2144-token attention block that the Qwen3.6 hybrid-mamba alignment needs, so the engine asserted at start-up.
- **Fix:** Set `--max-num-batched-tokens` to 4096. With the flag pinned, the AssertionError does not recur.
- **How we found it:** Hit during the procedure (step 4 of the source procedure); the 2144-token attention block vs. the 2048 the MTP path lowered the batched-tokens value to was identified from the launch failure and pinned per the stack table.

## 2. Switch script's resident process returns "fetch failed"

- **Symptom:** A resident switch process returns `fetch failed` against a backend that has just been restarted.
- **Root cause:** Stale connection held by the resident process; the restarted backend is up but the old socket/handle is dead.
- **Fix:** Restart the switch script.
- **How we found it:** Observed immediately after a backend restart; resolved by restarting the switch script. (The switch script's name, configuration and restart mechanics beyond this scrubbed symptom/fix line are not published.)

## 3. `scp` of adapters out of the training container fails with "Permission denied"

- **Symptom:** Copying LoRA adapter files out of the training container with `scp` produces root-owned files and a `Permission denied` error on the host side.
- **Root cause:** Files written inside the container are owned by root; the host user cannot read/move them as-is.
- **Fix:** `chown` the files first (inside the container, before the copy), then `scp`. The exact UID/GID to use for the serving user was not recorded; the symptom was read-side ownership mismatch visible in `ls -l`.
- **How we found it:** Hit while moving adapters from the training container to the host for serving; the ownership mismatch was visible from `ls -l`.

## 4. Short-`max_tokens` requests return empty `content`

- **Symptom:** Requests with a small `max_tokens` budget return an empty `content` field, as if the model produced nothing.
- **Root cause:** qwen thinking is on by default and consumes the entire token budget before any non-thinking output is emitted, so the visible `content` is empty.
- **Fix:** Force `enable_thinking=False`, passed as the `chat_template_kwargs` field on the request. This is also the first acceptance check in every inference script.
- **How we found it:** Thinking leakage was found at 7 sites; the empty-`content` pattern on short budgets traced back to thinking being on by default.

## 5. Vision-tower LoRA appears to do nothing

- **Symptom:** A LoRA adapter that includes vision-tower deltas (`visual.blocks.*`) appears to have no effect when served through vLLM.
- **Root cause:** vLLM silently ignores `visual.blocks.*` weights, logging "no matching PunicaWrapper". The language-side modules applied; the vision-tower modules were ignored by vLLM at that version.
- **Fix:** Keep visual-tower deltas out of vLLM-served LoRA expectations; do not rely on vLLM to apply them. Any reported score reflects the language-side adapter only.
- **How we found it:** A vision LoRA that should have changed image-side behavior did not; the "no matching PunicaWrapper" log line identified the silent-ignore path.

## 6. Host OOM / memory starvation

- **Symptom:** The host OOMs or shows memory starvation while other resident services on the host are running.
- **Root cause:** Another resident service on the host was over-allocating memory through a large context and high parallelism, starving the rest of the box.
- **Fix:** Cap that resident service's context and parallelism. This freed about 57 GB.
- **How we found it:** Host available RAM collapsed while the resident service was running; capping its context and parallelism recovered the headroom.

## 7. vLLM won't import on the GB10

- **Symptom:** vLLM fails to import on the GB10 (Grace, aarch64) host.
- **Root cause:** Wrong-architecture image tag — the default x86_64 tag on an aarch64 host breaks the Triton/PTXAS compile path, causing submodule import failures.
- **Fix:** Always use the `-aarch64-` suffixed tag (`vllm/vllm-openai:v0.24.0-aarch64-ubuntu2404`).
- **How we found it:** Hit on first pull of the default tag; the aarch64-suffixed tag imports cleanly.

## 8. `--memory` set at the machine's full RAM causes box OOM

- **Symptom:** Setting `--memory` to the machine's full RAM causes the box to OOM.
- **Root cause:** Other resident services on the host have no headroom and get killed when the container takes the full RAM budget.
- **Fix:** Leave host margin; here a hard cap below total RAM (55g).
- **How we found it:** Box OOM with the full-RAM cap; other resident services on the host were the victims, and a below-total cap restored headroom.

## 9. Serving via `PeftModel` (unmerged LoRA) on the base model is 20× slower

- **Symptom:** Inference served through `PeftModel` with an unmerged LoRA on the base model is dramatically slower than the merged-weights baseline.
- **Root cause:** `PeftModel` applies the LoRA delta on every forward pass instead of folding it into the base weights, adding a ~20× latency penalty.
- **Fix:** For evaluation, run the merged model; `PeftModel`-on-base inference was 20× slower in our runs (4 min vs 12 s per item). multi-LoRA serving in vLLM is a separate path.
- **How we found it:** Measured latency on the `PeftModel`-on-base path vs. the merged path; the 20× gap was reproducible.

## 10. Benchmark "regression" after switching serving stacks

- **Symptom:** A benchmark appears to regress after switching serving stacks (e.g. from vLLM to transformers, or between preprocessing paths).
- **Root cause:** The two evaluation conventions differ in serving stack and image preprocessing (forced square 768×768 resize vs aspect-ratio-preserving). The 66.67% ChartQA score came from a different item count that was not recorded, and is not comparable to the aspect-ratio-preserving vLLM protocol (87.5%).
- **Fix:** Compare only under identical preprocessing; never cite the 768×768-squash number as a model regression.
- **How we found it:** The 66.67% transformers-side score looked like a regression against the 87.5% vLLM-side score; the two evaluation conventions differ in serving stack and image preprocessing, and we did not isolate the resize alone, so the resize cannot be named as the sole cause.
