# Chosen method (v7): verbatim transcript ->
#   Qwen2.5-7B: rubric score + hidden-state embedding + minimal grammar-correction edit rate
#   Whisper-large-v3 encoder audio embedding + pause features (loudness-normalised)
#   ELECTRA-base student on gold labels (3 seeds, warm-up + linear decay)
# -> ridge stacker on out-of-fold predictions -> rule gate for unscorable audio (raw speech ratio ~1.0).
# Gold labels 0 (outside the 1-5 rubric) never train the scorer.
import glob, os, json, gc, re, time, difflib
import numpy as np, pandas as pd, torch, librosa
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import (AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification,
                          WhisperFeatureExtractor, WhisperModel)

SEED, T0 = 42, time.time()
SMOKE = os.environ.get("SMOKE") == "1"  # tiny stratified subset + 1 epoch/1 seed: catches bugs in minutes
OUT = os.environ.get("OUT_DIR", "/kaggle/working")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED); np.random.seed(SEED)
log = lambda *a: print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
root = os.environ.get("DATA_DIR") or os.path.dirname(glob.glob("/kaggle/input/**/train.csv", recursive=True)[0])
tr = pd.read_csv(os.environ.get("TRANSCRIPTS") or glob.glob("/kaggle/input/**/transcripts.csv", recursive=True)[0])
tr["transcript"] = tr.transcript.fillna("")
gold = pd.read_csv(f"{root}/train.csv")
df = tr[tr.split == "train"].merge(gold, on="filename").reset_index(drop=True)
te = tr[tr.split == "test"].reset_index(drop=True)
if SMOKE:
    df = pd.concat([g.head(12) for _, g in df.groupby(np.clip(np.round(df.label), 0, 5).astype(int))]).reset_index(drop=True)
    te = te.head(8)
alltext = pd.concat([df, te], ignore_index=True)
N = len(df)

# Same folds as v6 (same rows, bins, seed) so per-fold numbers are directly comparable
texts, ttexts, y = df.transcript.values, te.transcript.values, df.label.values
m1 = y >= 1
bins = np.where(m1, np.clip(np.round(y), 2, 5), 0).astype(int)  # zero clips get their own bin so every fold has ~7
folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(texts, bins))

# ---------- Stage 0: gate + transparent text features ----------
# Every train label-0 clip has raw speech ratio exactly 1.0 (loud throughout); the v6 logistic gate also fired on
# normal clips at 0.83-0.91. A rule on the raw ratio catches all zeros with no false alarms.
GATE = alltext.speech_ratio.values >= 0.99
FILL = r"\b(um+|uh+|hmm+|erm|er|ah)\b"
SUB = r"\b(because|which|who|whom|whose|that|when|while|although|though|if|unless|since|whereas|where|after|before|until)\b"
def feats(t, dur):
    w = re.findall(r"[A-Za-z']+", t.lower()); n = max(len(w), 1)
    sents = [s for s in re.split(r"[.!?]+", t) if s.strip()]
    return dict(f_words=len(w), f_wpm=len(w) / max(dur, 1) * 60,
                f_filler=len(re.findall(FILL, t.lower())) / n,
                f_repeat=sum(a == b for a, b in zip(w, w[1:])) / n,
                f_sentlen=len(w) / max(len(sents), 1),
                f_clauses=(len(re.findall(SUB, t.lower())) + len(sents)) / max(len(sents), 1))
F = pd.DataFrame([feats(t, d) for t, d in zip(alltext.transcript, alltext.duration)])
F["f_duration"] = alltext.duration.values

# ---------- Stage A: Qwen teacher score, embedding, grammar-correction edit rate ----------
LLM = os.environ.get("LLM", "Qwen/Qwen2.5-7B-Instruct")
RUBRIC = """You are an expert English language assessor. Rate the GRAMMAR of a candidate's spontaneous spoken English response (a verbatim ASR transcript of a 45-60 second answer; fillers, repetitions and false starts are kept) on this 1-5 rubric:
1: The speaker struggles with sentence structure and syntax; only limited control of simple grammatical structures and memorized sentence patterns.
2: Limited understanding of sentence structure and syntax. Uses simple structures but consistently makes basic grammatical mistakes; may leave sentences incomplete.
3: Decent grasp of sentence structure but makes errors in grammatical structure, OR decent grasp of grammar but errors in sentence structure.
4: Strong understanding of sentence structure and syntax; good control of grammar. Occasional minor errors that do not cause misunderstanding; most are self-corrected.
5: High grammatical accuracy and confident command of complex grammar; rarely makes noticeable mistakes and self-corrects when needed.
Judge grammar only, not pronunciation, accent, vocabulary range or content. Answer with a single digit 1-5."""
GEC_SYS = ("You correct grammar in transcripts of spontaneous spoken English. Fix only grammatical errors (verb forms, "
           "agreement, articles, prepositions, word order, missing words, incomplete sentences) with the fewest possible "
           "edits. Keep the speaker's words, meaning and style; do not rephrase or improve vocabulary. "
           "Output only the corrected text.")
tok = AutoTokenizer.from_pretrained(LLM); tok.padding_side = "left"
llm = AutoModelForCausalLM.from_pretrained(LLM, dtype=torch.float16 if DEV == "cuda" else torch.float32, device_map="auto" if DEV == "cuda" else None).eval()
QMID = llm.config.num_hidden_layers * 2 // 3  # a middle-late layer; the last layer is tuned for next-token prediction
digit_ids = [tok.encode(str(d), add_special_tokens=False)[0] for d in range(1, 6)]

def teacher(text):
    """Expected rubric digit, plus [decision-token state (mid, last layer), mean state of the plain transcript (mid)]."""
    msgs = [{"role": "system", "content": RUBRIC},
            {"role": "user", "content": f"Transcript:\n\"\"\"{text.strip() or '(no speech)'}\"\"\"\n\nGrammar score (1-5):"}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(llm.device)
    plain = tok(text.strip() or "(no speech)", return_tensors="pt").to(llm.device)
    with torch.no_grad():
        o = llm(**enc, output_hidden_states=True)
        hs = llm(**plain, output_hidden_states=True).hidden_states[QMID][0].mean(0)
    p = torch.softmax(o.logits[0, -1, digit_ids].float(), -1).cpu().numpy()
    emb = torch.cat([o.hidden_states[QMID][0, -1].to(hs.device), o.hidden_states[-1][0, -1].to(hs.device), hs])
    return float((p * np.arange(1, 6)).sum()), emb.float().cpu().numpy()

def strip_disfluency(t):  # fillers and immediate repeats are counted elsewhere; the edit rate should see grammar only
    t = re.sub(FILL + r",?\s*", "", t, flags=re.I)
    return re.sub(r"\b(\w+)(\s+\1\b)+", r"\1", t, flags=re.I).strip()

def gec_all(src, bs=16):
    order, out = np.argsort([len(t) for t in src]), [""] * len(src)
    for b in range(0, len(order), bs):
        idx = order[b:b + bs]
        prompts = [tok.apply_chat_template([{"role": "system", "content": GEC_SYS}, {"role": "user", "content": src[i] or "(no speech)"}],
                                           tokenize=False, add_generation_prompt=True) for i in idx]
        enc = tok(prompts, return_tensors="pt", padding=True).to(llm.device)
        mx = min(int(max(len(src[i].split()) for i in idx) * 1.6) + 32, 420)
        with torch.no_grad():
            g = llm.generate(**enc, max_new_tokens=mx, do_sample=False, temperature=None, top_p=None, top_k=None)
        for i, t in zip(idx, tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)):
            out[i] = t.strip()
        if b % (bs * 10) == 0: log(f"gec {b}/{len(src)}")
    return out

def edit_rate(a, b):
    a, b = re.findall(r"[a-z']+", a.lower()), re.findall(r"[a-z']+", b.lower())
    ops = difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
    return min(sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in ops if tag != "equal") / max(len(a), 1), 1.0)

tq = [teacher(t) for t in alltext.transcript]
alltext["teacher"] = [s for s, _ in tq]; EQ = np.stack([e for _, e in tq]); del tq
log("teacher + qwen embeddings done", EQ.shape)
src = [strip_disfluency(t) for t in alltext.transcript]
corr = gec_all(src)
alltext["gec_rate"] = [edit_rate(a, b) for a, b in zip(src, corr)]
alltext.assign(source=src, corrected=corr)[["split", "filename", "gec_rate", "source", "corrected"]].to_csv(f"{OUT}/gec_outputs.csv", index=False)
del llm; gc.collect(); torch.cuda.empty_cache() if DEV == "cuda" else None
log("gec done; mean edit rate", round(float(alltext.gec_rate.mean()), 3))

# ---------- Stage B: audio -- Whisper encoder embedding + loudness-normalised pause features ----------
WHISPER, SR = "openai/whisper-large-v3", 16000
fe = WhisperFeatureExtractor.from_pretrained(WHISPER)
wenc = WhisperModel.from_pretrained(WHISPER, dtype=torch.float16 if DEV == "cuda" else torch.float32).encoder.to(DEV).eval()
WMID = len(wenc.layers) // 2

def audio_emb(audio):
    pieces = [audio[i:i + 30 * SR] for i in range(0, len(audio), 30 * SR)]
    pieces = [p for p in pieces if len(p) >= SR] or pieces[:1]
    x = fe(pieces, sampling_rate=SR, return_tensors="pt").input_features.to(DEV, wenc.dtype)
    with torch.no_grad():
        hs = wenc(x, output_hidden_states=True).hidden_states
    vec = [torch.cat([hs[WMID][k, :n].mean(0), hs[-1][k, :n].mean(0)]).float().cpu().numpy()
           for k, n in enumerate(max(1, min(1500, int(np.ceil(len(p) / SR * 50)))) for p in pieces)]  # 50 encoder frames/s
    return np.average(vec, axis=0, weights=[len(p) for p in pieces])

def prosody(audio):
    """Voiced frames = RMS above 10% of the clip's own 95th percentile, so quiet recordings are not penalised."""
    rms = librosa.feature.rms(y=audio, frame_length=400, hop_length=160)[0]  # 25 ms frames, 10 ms hop
    v = rms > 0.1 * max(np.percentile(rms, 95), 1e-8)
    idx = np.flatnonzero(v)
    if len(idx) < 2: return dict(f_speech_ratio=0.0, f_pause_rate=0.0, f_pause_mean=0.0, f_voiced_sec=0.0)
    inner = v[idx[0]:idx[-1] + 1]
    edges = np.flatnonzero(np.diff(np.r_[1, inner.astype(int), 1]))  # silent runs start/end alternately
    runs = edges[1::2] - edges[::2]
    pauses = runs[runs >= 50]  # >= 0.5 s
    return dict(f_speech_ratio=float(v.mean()), f_pause_rate=len(pauses) / max(len(inner) / 6000, 1e-3),
                f_pause_mean=float(pauses.mean() / 100) if len(pauses) else 0.0, f_voiced_sec=float(inner.sum() / 100))

EA, P = [], []
for split, fn in zip(alltext.split, alltext.filename):
    audio, _ = librosa.load(f"{root}/{split}/{fn}" if fn.endswith(".wav") else f"{root}/{split}/{fn}.wav", sr=SR, mono=True)
    EA.append(audio_emb(audio)); P.append(prosody(audio))
EA = np.stack(EA); P = pd.DataFrame(P)
del wenc; gc.collect(); torch.cuda.empty_cache() if DEV == "cuda" else None
F = pd.concat([F, P], axis=1)
F["f_artic"] = F.f_words / np.maximum(F.f_voiced_sec, 1) * 60  # words per voiced minute
F["f_gec"] = alltext.gec_rate.values
BASE = ["f_words", "f_wpm", "f_filler", "f_repeat", "f_sentlen", "f_clauses", "f_speech_ratio", "f_duration"]
PAUSE = ["f_pause_rate", "f_pause_mean", "f_artic"]
log("audio done", EA.shape)

# ---------- Stage C: student trained on gold labels, 3 seeds ----------
BB, MAXLEN, BS = os.environ.get("BACKBONE", "google/electra-base-discriminator"), 256, 16
EPOCHS, LR = (1 if SMOKE else 10), 2e-5
SEEDS = [42] if SMOKE else [42, 7, 2024]  # v6: fold RMSE swung 0.63-0.88 with one seed
tk = AutoTokenizer.from_pretrained(BB)
enc = lambda t: {k: v.to(DEV) for k, v in tk(list(t), truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt").items()}

def predict(model, texts, bs=32):
    model.eval(); out = []
    with torch.no_grad(), torch.autocast(DEV, dtype=torch.float16, enabled=DEV == "cuda"):
        for b in range(0, len(texts), bs):
            out.append(model(**enc(texts[b:b + bs])).logits.squeeze(-1).float().cpu().numpy())
    return np.concatenate(out)

def train_student(texts, y, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = AutoModelForSequenceClassification.from_pretrained(BB, num_labels=1, dtype=torch.float32).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = EPOCHS * int(np.ceil(len(texts) / BS)); warm = max(1, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / max(steps - warm, 1))))
    y = np.asarray(y, dtype=np.float32)
    for ep in range(EPOCHS):  # fp32 training, as in the E1 reference run
        model.train(); perm = rng.permutation(len(texts))
        for b in range(0, len(perm), BS):
            idx = perm[b:b + BS]
            loss = ((model(**enc(texts[idx])).logits.squeeze(-1) - torch.tensor(y[idx]).to(DEV)) ** 2).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    return model

oof_student, test_student = np.zeros(N), np.zeros(len(te))
insample = np.zeros(N); incount = np.zeros(N)
for seed in SEEDS:
    for k, (tri, vai) in enumerate(folds):
        tri = tri[m1[tri]]  # gold labels 1-5 only
        m = train_student(texts[tri], y[tri], seed)
        pv = predict(m, texts[vai]); oof_student[vai] += pv / len(SEEDS)
        test_student += predict(m, ttexts) / (len(folds) * len(SEEDS))
        insample[tri] += predict(m, texts[tri]); incount[tri] += 1
        del m; gc.collect(); torch.cuda.empty_cache() if DEV == "cuda" else None
        log(f"seed {seed} fold {k} student rmse(1-5)", round(float(np.sqrt(np.mean((pv[m1[vai]] - y[vai][m1[vai]]) ** 2))), 3))

# ---------- Stage D: level-1 ridges on embeddings, then ridge stacker on OOF inputs ----------
ridge = lambda a: make_pipeline(StandardScaler(), RidgeCV(alphas=a))
def oof_ridge(Z, alphas):
    o = np.zeros(N)
    for tri, vai in folds:
        tri = tri[m1[tri]]; o[vai] = ridge(alphas).fit(Z[:N][tri], y[tri]).predict(Z[:N][vai])
    full = ridge(alphas).fit(Z[:N][m1], y[m1])
    return o, full.predict(Z[N:]), full.predict(Z[:N])
EMB_ALPHAS = np.logspace(1, 6, 26)
q_oof, q_te, q_in = oof_ridge(EQ, EMB_ALPHAS)
a_oof, a_te, a_in = oof_ridge(EA, EMB_ALPHAS)

Ftr, Fte = F.iloc[:N].reset_index(drop=True), F.iloc[N:].reset_index(drop=True)
COLS = {"student": (oof_student, test_student), "teacher": (alltext.teacher.values[:N], alltext.teacher.values[N:]),
        "qwen_emb": (q_oof, q_te), "audio_emb": (a_oof, a_te)}
COLS.update({c: (Ftr[c].values, Fte[c].values) for c in BASE + PAUSE + ["f_gec"]})
stack = lambda: ridge(np.logspace(-2, 3, 30))
def Xof(names, part): return np.column_stack([COLS[n][part] for n in names])
def oof_stack_of(names):
    Xtr, o = Xof(names, 0), np.zeros(N)
    for tri, vai in folds:  # same folds; stacker fits only on label>=1 training-fold clips
        tri = tri[m1[tri]]; o[vai] = stack().fit(Xtr[tri], y[tri]).predict(Xtr[vai])
    return o

FULL = ["student", "teacher", "qwen_emb", "audio_emb"] + BASE + PAUSE + ["f_gec"]
VARIANTS = {"v6 inputs (student+teacher+base feats)": ["student", "teacher"] + BASE, "FULL (chosen)": FULL,
            "FULL - qwen_emb": [c for c in FULL if c != "qwen_emb"], "FULL - audio_emb": [c for c in FULL if c != "audio_emb"],
            "FULL - gec": [c for c in FULL if c != "f_gec"], "FULL - pause": [c for c in FULL if c not in PAUSE]}
oofs = {n: oof_stack_of(c) for n, c in VARIANTS.items()}
oof_stack = oofs["FULL (chosen)"]
final_stack = stack().fit(Xof(FULL, 0)[m1], y[m1])
test_stack = final_stack.predict(Xof(FULL, 1))

# ---------- metrics ----------
def M(yv, p): return dict(pearson=round(float(pearsonr(yv, p)[0]), 4), spearman=round(float(spearmanr(yv, p)[0]), 4),
                          rmse=round(float(np.sqrt(np.mean((yv - p) ** 2))), 4))
clip = lambda p: np.clip(p, 1, 5)
gate_tr, gate_te = GATE[:N], GATE[N:]
gated = lambda p, g: np.where(g, 0.0, clip(p))
rows = {"Teacher raw (zero-shot)": alltext.teacher.values[:N], "Qwen embedding ridge": q_oof, "Audio embedding ridge": a_oof,
        "Student (ELECTRA-base, 3 seeds)": oof_student, **{f"Stack: {n}": p for n, p in oofs.items()}}
res = {n: {"label>=1": M(y[m1], clip(p)[m1]), "all clips (gated)": M(y, gated(p, gate_tr))} for n, p in rows.items()}
res["gate on train"] = dict(recall=round(float(gate_tr[~m1].mean()), 3), false_alarms_on_1to5=int(gate_tr[m1].sum()),
                            test_flagged=te.filename[gate_te].tolist())
ins = np.where(incount > 0, insample / np.maximum(incount, 1), np.nan)
COLS["student"], COLS["qwen_emb"], COLS["audio_emb"] = (ins, test_student), (q_in, q_te), (a_in, a_te)
res["TRAINING RMSE (in-sample, mandatory)"] = M(y[m1], clip(final_stack.predict(Xof(FULL, 0)[m1])))
COLS["student"], COLS["qwen_emb"], COLS["audio_emb"] = (oof_student, test_student), (q_oof, q_te), (a_oof, a_te)
res["per-fold RMSE (1-5)"] = {n: [round(float(np.sqrt(np.mean((p[v][m1[v]] - y[v][m1[v]]) ** 2))), 4) for _, v in folds] for n, p in oofs.items()}
res["stack coefficients (standardised)"] = {k: round(float(v), 4) for k, v in zip(FULL, final_stack[-1].coef_)}
res["feature correlation with label (label>=1)"] = {c: round(float(pearsonr(Ftr[c][m1], y[m1])[0]), 3) for c in BASE + PAUSE + ["f_gec"]}
d45 = df.duration.values < 50; q10 = Ftr.f_speech_ratio.values < 0.5
res["FULL rmse by group (1-5)"] = {g: round(float(np.sqrt(np.mean((clip(oof_stack) - y)[m1 & s] ** 2))), 4)
                                   for g, s in {"45s": d45, "60s": ~d45, "low voiced ratio": q10}.items()}
print(json.dumps(res, indent=2))
json.dump(res, open(f"{OUT}/chosen_results.json", "w"), indent=2)

oof = df[["filename", "label"]].copy()
for n, p in rows.items(): oof[n] = p
oof["gate"] = gate_tr; oof.to_csv(f"{OUT}/chosen_oof.csv", index=False)
inputs = alltext[["split", "filename"]].copy()  # every stacker input, so the stack can be re-fitted without a GPU
for n in COLS: inputs[n] = np.r_[COLS[n][0], COLS[n][1]]
inputs.to_csv(f"{OUT}/stack_inputs.csv", index=False)
pd.DataFrame({"filename": te.filename, "label": gated(test_stack, gate_te)}).to_csv(f"{OUT}/submission.csv", index=False)
log("done; test clips gated:", int(gate_te.sum()))
