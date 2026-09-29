# SHL Grammar Scoring — rubric-taught, gold-tuned scorer

Solution for the Kaggle competition **SHL Hiring Assessment 2026**: predict a 0–5 grammar score for 45–60 s spoken-English clips.

## Method

```
audio ─▶ [1] verbatim ASR ─▶ transcript ─┬─▶ [A] teacher: Qwen2.5-7B-Instruct reads the rubric → expected score (1–5)
                                          ├─▶ [B] student: ELECTRA-base fine-tuned on gold labels (lr 2e-5, 10 epochs)
                                          └─▶ [C] 8 transparent features (words, wpm, fillers, repeats, sentence length,
                                                    clauses/sentence, speech ratio, duration)
          [3] ridge stacker on out-of-fold student + teacher + features ─▶ [4] unscorable-audio gate (optional) ─▶ clip [0, 5]
```

- **[1] Verbatim ASR** (`asr/asr.py`): Whisper-large-v3 with a filler-only prompt so fillers, repetitions and errors are kept. Each clip is cut at its quietest point between 20 and 30 s so the prompt applies to every piece. Any piece that looks like a hallucination (a repetition loop, prompt text, or a compression ratio above 2.4) is decoded again without the prompt. An earlier prompt containing a full example sentence was copied into 4% of transcripts.
- **[A] Teacher**: the probability-weighted mean over the next-token digits 1–5. Continuous and deterministic.
- **[B] Student**: ELECTRA-base, the backbone the host paper found best, trained on the gold labels 1–5. The paper's pseudo-label training with clean-sample selection (Das, Kumar & Yadav, AACL-IJCNLP 2025, arXiv:2511.13152) was tested and dropped: at ~475 clips per fold, selecting the lowest-loss 30% collapsed to a constant predictor (r ≈ 0), and as a warm-up it hurt gold tuning. With the selection off (α = 1) the student beat its LLM teacher (r 0.555 vs 0.540), as the paper reports.
- **[3] Stacker**: `RidgeCV` fitted only on out-of-fold predictions.
- **[4] Gate**: a logistic model on speech ratio, duration and word count, which detects the unscorable clips labelled 0/0.5. It is written to a separate submission file and **off by default**.

Clips labelled 0 or 0.5 (outside the 1–5 rubric) are never used to train the scorer.

## Validation

5-fold stratified CV (seed 42). Every learned component (student, stacker, gate) is fitted inside the training folds. The output reports Pearson, Spearman and RMSE on labels ≥ 1, RMSE on all clips, per-fold RMSE, the mandatory in-sample training RMSE, stacker coefficients and ablations (teacher, calibrated teacher, student, full stack, stack + gate).

## Run (Kaggle, GPU T4 ×2, internet on)

```bash
kaggle kernels push -p asr     # -> transcripts.csv (~40 min)
kaggle kernels push -p train   # -> chosen_results.json, chosen_oof.csv, submission_gate_off.csv, submission_gate_on.csv
```

Change the `id` / `kernel_sources` user in both `kernel-metadata.json` files to your Kaggle username. Setting `SMOKE=1` runs `train.py` on a tiny stratified subset with 1 epoch per stage, as a quick end-to-end check. `DATA_DIR`, `TRANSCRIPTS`, `OUT_DIR`, `LLM` and `BACKBONE` override the defaults.

## Compliance

- **No competition data leaves Kaggle** (rule 4b): open-weight models only, no hosted APIs. No competition data in this repository.
- **Licences** (rule 6c): Whisper-large-v3 (MIT), Qwen2.5-7B-Instruct (Apache-2.0), DeBERTa-v3-large (MIT). CrisperWhisper was not used because of its non-commercial licence.
- No external data, and no hand-labelling of test clips.

## Results

_To be filled in from `chosen_results.json`._
