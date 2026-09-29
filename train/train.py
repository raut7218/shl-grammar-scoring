# Chosen method: verbatim transcript -> Qwen teacher -> DeBERTa student (B1 warm-up on teacher w/ clean-sample
# selection, B2 gold tuning) -> ridge stacker with transparent features -> optional unscorable-audio gate.
# Gold labels 0/0.5 never train the scorer (user rule). Folds identical to E1.
import glob, os, json, gc, re, copy, time
import numpy as np, pandas as pd, torch
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_predict
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

SEED, T0 = 42, time.time()
SMOKE = os.environ.get("SMOKE") == "1"  # tiny stratified subset + 1 epoch per stage: catches bugs in minutes
OUT = os.environ.get("OUT_DIR", "/kaggle/working")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
EP = 1 if SMOKE else 4
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

# ---------- Stage C: transparent features ----------
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
F["f_speech_ratio"], F["f_duration"] = alltext.speech_ratio.values, alltext.duration.values
FEATS = list(F.columns)

# ---------- Stage A: teacher (same prompt as E1) ----------
LLM = os.environ.get("LLM", "Qwen/Qwen2.5-7B-Instruct")
RUBRIC = """You are an expert English language assessor. Rate the GRAMMAR of a candidate's spontaneous spoken English response (a verbatim ASR transcript of a 45-60 second answer; fillers, repetitions and false starts are kept) on this 1-5 rubric:
1: The speaker struggles with sentence structure and syntax; only limited control of simple grammatical structures and memorized sentence patterns.
2: Limited understanding of sentence structure and syntax. Uses simple structures but consistently makes basic grammatical mistakes; may leave sentences incomplete.
3: Decent grasp of sentence structure but makes errors in grammatical structure, OR decent grasp of grammar but errors in sentence structure.
4: Strong understanding of sentence structure and syntax; good control of grammar. Occasional minor errors that do not cause misunderstanding; most are self-corrected.
5: High grammatical accuracy and confident command of complex grammar; rarely makes noticeable mistakes and self-corrects when needed.
Judge grammar only, not pronunciation, accent, vocabulary range or content. Answer with a single digit 1-5."""
tok = AutoTokenizer.from_pretrained(LLM)
llm = AutoModelForCausalLM.from_pretrained(LLM, dtype=torch.float16 if DEV == "cuda" else torch.float32, device_map="auto" if DEV == "cuda" else None).eval()
digit_ids = [tok.encode(str(d), add_special_tokens=False)[0] for d in range(1, 6)]
def teacher(text):
    msgs = [{"role": "system", "content": RUBRIC},
            {"role": "user", "content": f"Transcript:\n\"\"\"{text.strip() or '(no speech)'}\"\"\"\n\nGrammar score (1-5):"}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(llm.device)
    with torch.no_grad():
        p = torch.softmax(llm(**enc).logits[0, -1, digit_ids].float(), -1).cpu().numpy()
    return float((p * np.arange(1, 6)).sum())
alltext["teacher"] = [teacher(t) for t in alltext.transcript]
del llm; gc.collect(); torch.cuda.empty_cache() if DEV == "cuda" else None
df["teacher"], te["teacher"] = alltext.teacher.values[:len(df)], alltext.teacher.values[len(df):]
alltext[["split","filename","teacher"]].to_csv(f"{OUT}/teacher_scores.csv", index=False)
log("teacher done")

# ---------- Stage B: DeBERTa student ----------
BB, MAXLEN, BS = os.environ.get("BACKBONE", "microsoft/deberta-v3-large"), 256, 8
tk = AutoTokenizer.from_pretrained(BB)
enc = lambda t: {k: v.to(DEV) for k, v in tk(list(t), truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt").items()}

def predict(model, texts, bs=32):
    model.eval(); out = []
    with torch.no_grad(), torch.autocast(DEV, dtype=torch.float16, enabled=DEV == "cuda"):
        for b in range(0, len(texts), bs):
            out.append(model(**enc(texts[b:b + bs])).logits.squeeze(-1).float().cpu().numpy())
    return np.concatenate(out)

def optimizer(model, lr, decay):
    # layer-wise lr decay: head gets lr, each encoder layer below gets lr * decay^depth, embeddings the lowest
    layers = model.deberta.encoder.layer; L = len(layers); groups = []
    groups.append({"params": [p for n, p in model.named_parameters() if not n.startswith("deberta.")], "lr": lr})
    groups.append({"params": list(model.deberta.encoder.rel_embeddings.parameters()) +
                   (list(model.deberta.encoder.LayerNorm.parameters()) if hasattr(model.deberta.encoder, "LayerNorm") else []), "lr": lr})
    for i, layer in enumerate(layers):
        groups.append({"params": list(layer.parameters()), "lr": lr * decay ** (L - i)})
    groups.append({"params": list(model.deberta.embeddings.parameters()), "lr": lr * decay ** (L + 1)})
    return torch.optim.AdamW(groups, weight_decay=0.01)

def run_epochs(model, texts, y, epochs, lr, alpha=1.0, decay=1.0, val=None):
    opt, scaler = optimizer(model, lr, decay), torch.amp.GradScaler(DEV, enabled=DEV == "cuda")
    y = np.asarray(y, dtype=np.float32); active = np.arange(len(texts)); best = (np.inf, None)
    for ep in range(epochs):
        model.train(); perm = np.random.permutation(active)
        for b in range(0, len(perm), BS):
            idx = perm[b:b + BS]
            with torch.autocast(DEV, dtype=torch.float16, enabled=DEV == "cuda"):
                pred = model(**enc(texts[idx])).logits.squeeze(-1)
            loss = ((pred.float() - torch.tensor(y[idx]).to(DEV)) ** 2).mean()
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update()
        if alpha < 1:  # author's clean-sample selection: keep lowest-loss alpha fraction for next epoch
            active = np.argsort((predict(model, texts) - y) ** 2)[: int(alpha * len(texts))]
        if val is not None:  # pick best epoch on inner split
            r = float(np.sqrt(np.mean((predict(model, val[0]) - val[1]) ** 2)))
            if r < best[0]: best = (r, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    if val is not None: model.load_state_dict(best[1])
    return model

def new_model():
    return AutoModelForSequenceClassification.from_pretrained(BB, num_labels=1, dtype=torch.float32).to(DEV)  # v5 keeps checkpoint dtype (fp16) otherwise

texts, ttexts, y = df.transcript.values, te.transcript.values, df.label.values
m1 = y >= 1
bins = np.clip(np.round(y), 2, 5).astype(int)
folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(texts, bins))  # identical to E1
V = {k: np.zeros(len(df)) for k in ["B1", "B2only", "B1B2"]}
T = {k: np.zeros(len(te)) for k in V}
insample = np.zeros(len(df)); incount = np.zeros(len(df))
for k, (tri, vai) in enumerate(folds):
    tri = tri[m1[tri]]
    inner_tr, inner_va = train_test_split(tri, test_size=0.1, random_state=SEED, stratify=bins[tri])
    val = (texts[inner_va], y[inner_va])
    # B1: warm-up on teacher scores (no gold)
    m = run_epochs(new_model(), texts[tri], df.teacher.values[tri], epochs=EP, lr=1e-5, alpha=0.3)
    V["B1"][vai] = predict(m, texts[vai]); T["B1"] += predict(m, ttexts) / 5
    # B2 after B1: gold tuning with LLRD, best epoch on inner split
    m = run_epochs(m, texts[inner_tr], y[inner_tr], epochs=EP, lr=8e-6, decay=0.9, val=val)
    V["B1B2"][vai] = predict(m, texts[vai]); T["B1B2"] += predict(m, ttexts) / 5
    insample[tri] += predict(m, texts[tri]); incount[tri] += 1
    del m; gc.collect(); torch.cuda.empty_cache() if DEV == "cuda" else None
    # Ablation: B2 only (no warm-up)
    m = run_epochs(new_model(), texts[inner_tr], y[inner_tr], epochs=EP, lr=8e-6, decay=0.9, val=val)
    V["B2only"][vai] = predict(m, texts[vai]); T["B2only"] += predict(m, ttexts) / 5
    del m; gc.collect(); torch.cuda.empty_cache() if DEV == "cuda" else None
    log(f"fold {k} done", {n: round(float(np.sqrt(np.mean((V[n][vai][m1[vai]] - y[vai][m1[vai]]) ** 2))), 3) for n in V})

# ---------- Stage 3: ridge stacker on OOF inputs ----------
def X(frame_student, frame_teacher, Fpart):
    return np.column_stack([frame_student, frame_teacher, Fpart[FEATS].values])
Ftr, Fte = F.iloc[:len(df)].reset_index(drop=True), F.iloc[len(df):].reset_index(drop=True)
stack = lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30)))
Xtr, Xte = X(V["B1B2"], df.teacher.values, Ftr), X(T["B1B2"], te.teacher.values, Fte)
oof_stack = np.zeros(len(df))
for tri, vai in folds:  # same folds; stacker fits only on label>=1 training-fold clips
    tri = tri[m1[tri]]; oof_stack[vai] = stack().fit(Xtr[tri], y[tri]).predict(Xtr[vai])
final_stack = stack().fit(Xtr[m1], y[m1])
test_stack = final_stack.predict(Xte)
coefs = dict(zip(["student", "teacher"] + FEATS, np.round(final_stack[-1].coef_, 4)))
# teacher-only stack (calibrated teacher) and teacher+features, for the ablation table
def oof_of(cols):
    o = np.zeros(len(df))
    for tri, vai in folds:
        tri = tri[m1[tri]]; o[vai] = stack().fit(cols[tri], y[tri]).predict(cols[vai])
    return o
oof_teacher_cal = oof_of(df.teacher.values[:, None])

# ---------- Stage 4: unscorable-audio gate (reported; OFF by default) ----------
G = Ftr[["f_speech_ratio", "f_duration", "f_words"]].values
gate = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced"))
p_zero = np.zeros(len(df))
for tri, vai in folds:
    p_zero[vai] = gate.fit(G[tri], (y[tri] < 1).astype(int)).predict_proba(G[vai])[:, 1]
ZERO = float(y[~m1].mean())
gated = np.where(p_zero > 0.5, ZERO, oof_stack)
gate.fit(G, (y < 1).astype(int))
test_gate = gate.predict_proba(Fte[["f_speech_ratio", "f_duration", "f_words"]].values)[:, 1]

# ---------- metrics ----------
def M(yv, p): return dict(pearson=round(float(pearsonr(yv, p)[0]), 4), spearman=round(float(spearmanr(yv, p)[0]), 4),
                          rmse=round(float(np.sqrt(np.mean((yv - p) ** 2))), 4))
clip = lambda p: np.clip(p, 0, 5)
rows = {"Teacher raw (zero-shot)": df.teacher.values, "Teacher + linear calibration": oof_teacher_cal,
        "B1 student (author warm-up only)": V["B1"], "B2 only (gold, no warm-up)": V["B2only"],
        "B1 + B2 (chosen student)": V["B1B2"], "Full stack (chosen)": oof_stack}
res = {n: {"label>=1": M(y[m1], clip(p)[m1]), "all clips": M(y, clip(p))} for n, p in rows.items()}
res["Full stack + gate (all clips)"] = {"all clips": M(y, clip(gated))}
zi = ~m1
res["gate detection on 0/0.5 clips"] = dict(recall=round(float((p_zero[zi] > .5).mean()), 3),
                                            false_alarms_on_1to5=int((p_zero[m1] > .5).sum()))
ins = np.where(incount > 0, insample / np.maximum(incount, 1), np.nan)
Xin = X(ins, df.teacher.values, Ftr)
res["TRAINING RMSE (in-sample, mandatory)"] = M(y[m1], clip(final_stack.predict(Xin[m1])))
res["per-fold RMSE full stack"] = [round(float(np.sqrt(np.mean((oof_stack[v][m1[v]] - y[v][m1[v]]) ** 2))), 4) for _, v in folds]
res["stack coefficients (standardised)"] = {k: float(v) for k, v in coefs.items()}
res["feature correlation with label (label>=1)"] = {c: round(float(pearsonr(Ftr[c][m1], y[m1])[0]), 3) for c in FEATS}
print(json.dumps(res, indent=2))
json.dump(res, open(f"{OUT}/chosen_results.json", "w"), indent=2)

oof = df[["filename", "label", "teacher"]].copy()
for n, p in rows.items(): oof[n] = p
oof["p_zero"] = p_zero; oof.to_csv(f"{OUT}/chosen_oof.csv", index=False)
# Two candidate submissions, NOT submitted: gate off (default) and gate on
pd.DataFrame({"filename": te.filename, "label": clip(test_stack)}).to_csv(f"{OUT}/submission_gate_off.csv", index=False)
pd.DataFrame({"filename": te.filename, "label": clip(np.where(test_gate > .5, ZERO, test_stack))}).to_csv(f"{OUT}/submission_gate_on.csv", index=False)
log("done; test clips flagged by gate:", int((test_gate > .5).sum()))
