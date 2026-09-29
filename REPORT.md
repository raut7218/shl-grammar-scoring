# Report — Spoken grammar scoring for the SHL Hiring Assessment 2026

This report explains the whole system for review: what happens to one audio clip from start to finish, why each stage exists, how the model is trained and validated, exactly how the metrics are computed, and what was checked for leakage and bugs.

---

## 1. The task in one paragraph

Each input is a 45–60 second WAV of a candidate speaking English spontaneously. The output is a grammar score from 0 to 5. Training labels are human ratings on a 1–5 rubric in 0.5 steps, plus 37 clips labelled 0 that the rubric does not describe. There are 769 labelled training clips and 216 test clips. Grammar is carried by the words and their order, so the system turns speech into a **verbatim** transcript and scores the transcript, with a small amount of help from audio-level measurements.

---

## 2. The pipeline, following one clip

```
 audio_123.wav (45–60 s)
      │
      ▼
 ┌──────────────────────────────────────────────────────────────┐
 │ STAGE 1 · Verbatim ASR                       asr/asr.py      │
 │  load 16 kHz mono → measure duration + speech ratio          │
 │  cut at quietest point in 20–30 s → pieces ≤ 30 s            │
 │  Whisper-large-v3 + filler-only prompt → text per piece      │
 │  hallucination check → re-decode flagged piece w/o prompt    │
 │  join pieces → transcripts.csv                               │
 └──────────────────────────────────────────────────────────────┘
      │  transcript, duration, speech_ratio
      ├──────────────────────┬──────────────────────┐
      ▼                      ▼                      ▼
 ┌────────────────┐  ┌──────────────────┐  ┌─────────────────────┐
 │ A · Teacher    │  │ B · Student      │  │ C · 8 features      │
 │ Qwen2.5-7B     │  │ ELECTRA-base     │  │ words, wpm, fillers,│
 │ reads rubric → │  │ fine-tuned on    │  │ repeats, sentence   │
 │ P(1..5) →      │  │ gold labels 1–5  │  │ length, clauses,    │
 │ expected score │  │ → score          │  │ speech ratio, dur.  │
 └────────────────┘  └──────────────────┘  └─────────────────────┘
      │                      │                      │
      └──────────────────────┴──────────────────────┘
                             ▼
 ┌──────────────────────────────────────────────────────────────┐
 │ STAGE 3 · Ridge stacker (10 inputs → 1 score)                │
 └──────────────────────────────────────────────────────────────┘
                             ▼
 ┌──────────────────────────────────────────────────────────────┐
 │ STAGE 4 · Unscorable-audio gate                              │
 │  P(clip looks like the label-0 clips) > 0.5 → output 0       │
 │  otherwise keep the stacker's score; clip to [0, 5]          │
 └──────────────────────────────────────────────────────────────┘
                             ▼
                     final score, e.g. 3.42
```

### Stage 1 — Verbatim speech recognition (`asr/asr.py`)

| Step | What happens | Why |
|---|---|---|
| Load | `librosa.load(sr=16000, mono=True)` | Whisper expects 16 kHz mono |
| Measure | `duration` in seconds. `speech_ratio` = share of 128 ms frames (32 ms hop, librosa defaults) whose RMS energy exceeds 0.02 | Used later as features and by the gate |
| Split | While the audio is over 30 s: find the quietest 50 ms frame between 20 s and 30 s and cut there | Whisper handles 30 s windows. Cutting at a quiet point avoids splitting a word, and every piece gets the prompt |
| Transcribe | Whisper-large-v3, English, greedy decoding, prompt `"Um, uh, so, like, I I mean, uh, you know, hmm."` | Most ASR "cleans" speech by deleting fillers and repairing errors, which erases what the rubric scores. A disfluent prompt makes Whisper keep them |
| Check | A piece is flagged if a 2–6 word phrase repeats 4 times in a row, its gzip compression ratio exceeds 2.4 (Whisper's own hallucination heuristic), or it contains the prompt text | An earlier prompt with a full example sentence was pasted into 4% of transcripts. The check catches that failure |
| Retry | Flagged pieces are decoded again without the prompt | A prompt-free decode does not hallucinate the prompt |
| Join | Pieces are concatenated per clip | → `transcripts.csv` (split, filename, transcript, duration, speech_ratio, n_words, flagged) |

On the final run, 8 of 2,583 pieces were retried, and no clip remained flagged.

### Stage A — Teacher: an LLM that reads the rubric

Qwen2.5-7B-Instruct receives the competition rubric (bands 1–5) as a system message and the transcript as the user message, ending with `Grammar score (1-5):`. Nothing is generated. One forward pass gives the model's next-token scores for the tokens `1` … `5`, and a softmax turns them into probabilities:

```
teacher = Σ_{d=1..5}  d · P(next token = d)          e.g. 0.05·2 + 0.60·3 + 0.35·4 = 3.30
```

This gives a continuous, deterministic score from the model's rubric knowledge. It never sees a label, so it cannot leak.

### Stage B — Student: ELECTRA-base fine-tuned on the gold labels

`google/electra-base-discriminator` with a one-output regression head. Input: the transcript, truncated to 256 tokens (enough for about 130 words). Training: MSE loss, AdamW, lr 2e-5, weight decay 0.01, batch 16, 10 epochs, fp32. Clips labelled 0 are never used for training. Five models are trained, one per cross-validation fold (Section 3). Each predicts the fold it did not see, and the five are averaged for the test clips.

### Stage C — Eight transparent features

| Feature | Definition |
|---|---|
| `f_words` | Number of words |
| `f_wpm` | Words per minute of audio |
| `f_filler` | (um, uh, hmm, er, ah) per word |
| `f_repeat` | Immediately repeated words ("the the") per word |
| `f_sentlen` | Words per sentence (split on `. ! ?`) |
| `f_clauses` | (subordinators such as *because, which, when, although* + sentences) per sentence: a complexity proxy |
| `f_speech_ratio` | From Stage 1 |
| `f_duration` | From Stage 1 |

### Stage 3 — Ridge stacker

`StandardScaler → RidgeCV(alphas = 10^-2 … 10^3)` over 10 inputs: student, teacher and the 8 features. With about 730 labelled clips, a linear model is the right size. Its standardised coefficients show how much each input drives the final score (Section 5).

### Stage 4 — Unscorable-audio gate

The 37 clips labelled 0 contain normal English speech but are loud from start to finish (median speech ratio 1.00, against 0.53–0.62 for every other band). This looks like background noise or broken recordings, not bad grammar. A logistic regression on `(speech_ratio, duration, n_words)` with balanced class weights predicts P(unscorable). Above 0.5, the output is the mean label of the zero clips (0.0). Otherwise the stacker's score is kept. Finally, everything is clipped to [0, 5].

---

## 3. Training and validation

**Five-fold stratified cross-validation, seed 42.** Each clip gets a stratum: its label rounded and clipped to 2–5 (the one 1.0 and three 1.5 labels merge into 2), or a separate stratum 0 for the zero clips so that every fold holds about 7 of them. Every clip lands in exactly one validation fold.

For each fold *k*:

1. **Student.** Train on the fold's training clips with labels ≥ 1, then predict the validation clips (these become the *out-of-fold*, OOF, student scores) and all test clips.
2. **Stacker (OOF).** Fit on the training clips' OOF student scores + teacher + features, then predict the validation clips.
3. **Gate (OOF).** Fit on all training clips (target = label is 0), then predict the validation clips. The replacement value is the mean zero label of the training clips only.

After the loop, a final stacker is fitted on all OOF rows with labels ≥ 1, and a final gate on all training clips. Both are applied to the test clips' teacher score, 5-model-average student score and features.

**What "out-of-fold" guarantees:** every validation number is a prediction for a clip whose label the student, stacker and gate for that fold never saw.

---

## 4. How the metrics are computed

For true labels `y` and predictions `p` over *n* clips (predictions are clipped to [0, 5] first):

```
RMSE    = sqrt( (1/n) · Σ (y_i − p_i)² )                          # in score points; lower is better
Pearson = Σ (y_i − ȳ)(p_i − p̄) / sqrt( Σ (y_i − ȳ)² · Σ (p_i − p̄)² )  # −1..1; higher is better
Spearman = Pearson computed on the ranks of y and p
```

In code (`train.py`, function `M`): `np.sqrt(np.mean((y - p) ** 2))` and `scipy.stats.pearsonr / spearmanr`.

Reported on two populations:

- **labels 1–5** (732 clips): the grammar model's quality.
- **all clips** (769): what the leaderboard sees, since the test set presumably contains unscorable clips too.

**Cross-check (done for this report):** from the saved OOF file, Pearson was recomputed by hand from the formula above and with `numpy.corrcoef`, and RMSE by hand and with `sklearn.metrics.mean_squared_error`. All agree with the pipeline to 4 decimals for every model. Sanity check: predicting the training mean for every clip gives RMSE 1.2382, exactly the label standard deviation, as it must.

**Mandatory training RMSE.** The rules ask for the RMSE on the training data. Each training clip is scored by the 4 fold-models that *did* train on it (in-sample), and those scores go through the final stacker. This number is optimistic by design. The OOF number is the honest estimate of test performance.

**The leaderboard metric.** Kaggle names it "Pearson Correlation and RMSE" without a formula. The board ranks lower as better, and predicting the mean scores about 1.22, so it behaves like an RMSE. Our public score (0.4577) is lower than our OOF RMSE, so the formula probably mixes in the Pearson term. The OOF numbers here are the ones we can compute exactly.

---

## 5. Results (out-of-fold, 769 training clips)

| Model | Pearson (1–5) | Spearman (1–5) | RMSE (1–5) | RMSE (all clips) |
|---|---|---|---|---|
| Predict the training mean (baseline) | — | — | — | 1.238 |
| Teacher, zero-shot (Qwen2.5-7B) | 0.582 | 0.628 | 0.847 | 1.012 |
| Teacher + linear calibration | 0.581 | 0.570 | 0.825 | 0.954 |
| Student (ELECTRA-base, gold) | 0.793 | 0.799 | 0.727 | 0.999 |
| **Full stack** | **0.805** | **0.803** | **0.601** | 0.836 |
| **Full stack + gate** | — | — | — | **0.603** (Pearson 0.874) |

- **Per-fold RMSE** of the full stack (labels 1–5): 0.598, 0.621, 0.574, 0.647, 0.563. The folds are stable.
- **Training RMSE (in-sample, mandatory): 0.234**, against 0.601 out-of-fold. The student memorises the clips it trained on. The OOF figure is the honest estimate.
- **Gate:** out of fold it catches 37 of 37 zero clips, with 2 false alarms among 732 normal clips. On test it flags 4 of 216 clips. Without the gate, each zero clip is scored around 3, which is why "all clips" jumps from 0.603 to 0.836.
- **What drives the score** (standardised ridge coefficients): student 0.68, word count 0.35, words-per-minute −0.28, teacher 0.16, duration −0.11, speech ratio 0.07, the others ≈ 0. More words in the same time and a strong text-model score raise the grammar score. The LLM teacher adds signal the student lacks. The negative weight on words-per-minute counterbalances word count: fast but short answers are not rewarded.
- **Public leaderboard:** the first submission (the previous run, before the two audit fixes) scored **0.4577**, 2nd of 5 at the time.

---

## 6. Leakage and correctness audit

| # | Check | Finding |
|---|---|---|
| 1 | Metric formulas | ✅ Recomputed three independent ways; identical to 4 decimals |
| 2 | Student | ✅ Each fold model trains only on its training clips (labels ≥ 1) and predicts the unseen fold |
| 3 | Teacher and features | ✅ They use no labels, so they cannot leak |
| 4 | Train/test filename overlap | ✅ 212 names exist in both folders but are **different recordings** (transcript similarity ≤ 0.20, median 0.03). All joins use (split, filename) |
| 5 | Zero clips across folds | 🐞 **Fixed.** They were stratified together with the 2s, giving 6–10 per fold. They now have their own stratum |
| 6 | Gate replacement value | 🐞 **Fixed.** It was the mean over all clips, i.e. including the validation fold. It is now computed from the training folds only. No numeric effect, since every zero label is exactly 0.0 |
| 7 | Stacker inputs | ⚠️ Standard stacking caveat. For validation fold *k*, the stacker's training rows carry OOF student scores produced by models that saw fold *k*'s labels. This second-order effect is accepted Kaggle practice and small |
| 8 | Model selection | ⚠️ ELECTRA-base and its settings were chosen by comparing CV results on the same folds, so CV is slightly optimistic. There is no untouched holdout; the leaderboard plays that role |
| 9 | Test-time averaging | ⚠️ The stacker is fitted on single-model OOF student scores but applied to a 5-model average on test (slightly less spread). Standard practice |
| 10 | Submission format | ✅ `sample_submission.csv` is stale (only 25 of its 204 names exist in the test folder). Submissions use the 216 names in `test.csv`, which match the test audio exactly, and Kaggle accepted them |

---

## 7. What was tried and dropped

- **Host paper's method** (Das, Kumar & Yadav, AACL-IJCNLP 2025): LLM pseudo-labels, then a transformer trained each epoch on only the 30% of samples with the lowest loss. At ~475 clips per fold, this collapses to a constant predictor (Pearson ≈ 0) for ELECTRA-base, ELECTRA-large and BERT-base. Used as a warm-up before gold tuning, it made the result worse (RMSE 0.864 vs 0.783). Without the selection, the student slightly beat its teacher (r 0.555 vs 0.540), which matches the paper's claim.
- **DeBERTa-v3-large** as the student (lr 8e-6, 4 epochs, layer-wise decay): RMSE 0.783, worse than ELECTRA-base.
- **Whisper prompt with a full example sentence**: copied into 4% of transcripts. Replacing it with a filler-only prompt improved the full stack from RMSE 0.638 to 0.602.

## 8. Compliance

- Open-weight models only, run inside Kaggle notebooks. No competition data goes to any outside service (rule 4b).
- Licences: Whisper-large-v3 MIT, Qwen2.5-7B-Instruct Apache-2.0, ELECTRA Apache-2.0 (rule 6c). CrisperWhisper was not used because both versions carry non-commercial licences.
- No external data, and no hand-labelling of test clips.
