# SHL Grammar Scoring — text + audio stacked scorer

Solution for the Kaggle competition **SHL Hiring Assessment 2026**: predict a 0–5 grammar score for 45–60 s spoken-English clips.

## Method (v7)

```
audio ─▶ [1] verbatim ASR (Whisper-large-v3) ─▶ transcript ─┬─▶ [A] Qwen2.5-7B-Instruct: rubric score, hidden-state embedding,
   │                                                         │        minimal grammar-correction edit rate
   │                                                         ├─▶ [C] student: ELECTRA-base on gold labels, 3 seeds
   │                                                         └─▶ [D] transparent text features
   └─▶ [B] Whisper-large-v3 encoder embedding + loudness-normalised pause features
          [E] ridges on the two embeddings (out-of-fold) ─▶ ridge stacker on all OOF inputs ─▶ [F] rule gate ─▶ clip [1, 5]
```

- **[1] Verbatim ASR** (`asr/asr.py`): Whisper-large-v3 with a filler-only prompt, so fillers, repetitions and errors are kept.
  - Each clip is cut at its quietest point between 20 and 30 s, so the prompt applies to every piece.
  - Any piece that looks like a hallucination (a repetition loop, prompt text, or a compression ratio above 2.4) is decoded again without the prompt.
- **[A] Qwen2.5-7B-Instruct** (`train/train.py`), three signals from one model:
  - **Teacher:** the probability-weighted mean over the next-token digits 1–5 under the rubric prompt.
  - **Embedding:** the rubric-prompt decision-token state (a middle layer and the last layer), plus the mean state of the plain transcript. A ridge regression turns it into a score.
  - **Grammar-correction edit rate:** Qwen corrects the transcript with minimal edits, after fillers and immediate repeats are stripped. The feature is the share of words changed.
- **[B] Audio:**
  - **Embedding:** the Whisper-large-v3 encoder output, mean-pooled over the speech frames (a middle layer and the last layer). A ridge regression turns it into a score.
  - **Pause features:** pause rate, mean pause length and words per voiced minute. Voiced frames are those above 10% of the clip's own 95th-percentile loudness, so quiet recordings are not penalised.
- **[C] Student:** ELECTRA-base trained on the gold labels 1–5 (lr 2e-5, 10 epochs, 10% warm-up, then linear decay, gradient clipping at 1.0). 3 seeds per fold, averaged.
- **[D] Features:** words, words per minute, fillers, repeats, sentence length, clauses per sentence, loudness-normalised speech ratio, duration.
- **[E] Stacker:** `RidgeCV` fitted only on out-of-fold inputs.
- **[F] Gate:** every training clip labelled 0 has a raw speech ratio of exactly 1.0 (loud from start to finish). A clip is scored 0 when its raw ratio is ≥ 0.99. v6 used a logistic gate, which also fired on normal clips at 0.83–0.91.

The 37 clips labelled 0 (outside the 1–5 rubric) are never used to train the scorer.

## Validation

5-fold stratified CV (seed 42), with the same folds as v6. Every learned component (student, embedding ridges, stacker) is fitted inside the training folds. `chosen_results.json` reports:
- Pearson, Spearman and RMSE on labels ≥ 1, and on all clips with the gate applied;
- per-fold RMSE;
- the mandatory in-sample training RMSE;
- the stacker coefficients;
- a leave-one-block-out ablation;
- RMSE on 45 s, 60 s and quiet clips.

`stack_inputs.csv` holds every stacker input for train and test, so the stack can be re-fitted without a GPU.

## Results (5-fold out-of-fold, 769 training clips)

| Model | Pearson (1–5) | RMSE (1–5) | RMSE (all clips, gated) |
|---|---|---|---|
| Teacher, zero-shot (Qwen2.5-7B) | 0.582 | 0.847 | 0.827 |
| Qwen embedding ridge | 0.804 | 0.603 | 0.589 |
| Student (ELECTRA-base, 3 seeds) | 0.811 | 0.705 | 0.688 |
| Audio embedding ridge (Whisper encoder) | 0.844 | 0.544 | 0.530 |
| Stack with v6 inputs | 0.820 | 0.580 | 0.566 |
| **Full stack (submitted)** | **0.861** | **0.516** | **0.504** |

Leave-one-block-out (RMSE on labels 1–5):

| Block removed | RMSE |
|---|---|
| Audio embedding | 0.576 |
| Qwen embedding | 0.516 |
| Grammar-correction edit rate | 0.517 |
| Pause features | 0.514 |

The audio embedding accounts for almost all of the gain over v6 (0.601).

- **Per-fold RMSE:** 0.538, 0.497, 0.508, 0.561, 0.478. v6: 0.598, 0.621, 0.574, 0.647, 0.563.
- **Largest stacker coefficients (standardised):** audio embedding 0.48, student 0.31, words 0.11, words per minute −0.08, Qwen embedding 0.06.
- **Gate:** catches 37/37 zero clips with 0 false alarms. It flags 1 of 216 test clips.
- **Training RMSE** (in-sample, mandatory): 0.264.
- **Public leaderboard:** 0.3548 (v5 was 0.4577).

## Run (Kaggle, GPU T4 ×2, internet on)

```bash
kaggle kernels push -p asr     # -> transcripts.csv (~25 min)
kaggle kernels push -p train   # -> chosen_results.json, chosen_oof.csv, stack_inputs.csv, gec_outputs.csv, submission.csv (~85 min)
```

Change the `id` / `kernel_sources` user in both `kernel-metadata.json` files to your Kaggle username.
- `SMOKE=1` runs `train.py` on a tiny stratified subset with 1 epoch and 1 seed (~5 min end to end).
- `DATA_DIR`, `TRANSCRIPTS`, `OUT_DIR`, `LLM` and `BACKBONE` override the defaults.

## Compliance

- **No competition data leaves Kaggle** (rule 4b): open-weight models only, no hosted APIs. No competition data in this repository.
- **Licences** (rule 6c): Whisper-large-v3 (MIT), Qwen2.5-7B-Instruct (Apache-2.0), ELECTRA (Apache-2.0). CrisperWhisper was not used because of its non-commercial licence.
- No external data, and no hand-labelling of test clips.

[REPORT.md](REPORT.md) holds the detailed v6 architecture write-up, the metric definitions and the leakage audit.
