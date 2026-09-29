# Stage 1: verbatim ASR for every train+test clip -> transcripts.csv
import glob, os, time, zlib
import pandas as pd, numpy as np, torch, librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration

MODEL = "openai/whisper-large-v3"  # MIT licence (CrisperWhisper v1/v2 are non-commercial -> break the OSI rule)
# Filler-only prompt nudges Whisper to keep fillers and repeats. v1 used a full ungrammatical sentence, which Whisper
# pasted into ~4% of transcripts ("I go to the office..."), so the prompt now holds no copyable content.
PROMPT = "Um, uh, so, like, I I mean, uh, you know, hmm."
SR, BS = 16000, 16
SMOKE = os.environ.get("SMOKE") == "1"

root = os.path.dirname(glob.glob("/kaggle/input/**/train.csv", recursive=True)[0])
rows = [("train", f) for f in pd.read_csv(f"{root}/train.csv").filename] + \
       [("test", f) for f in pd.read_csv(f"{root}/test.csv").filename]
if SMOKE:
    rows = rows[:12] + rows[-8:]

def split30(audio):
    """Cut into <=30 s pieces at the quietest 50 ms frame between 20 s and 30 s, so prompts apply to every piece."""
    pieces = []
    while len(audio) > 30 * SR:
        seg = audio[20 * SR:30 * SR]
        rms = librosa.feature.rms(y=seg, frame_length=800, hop_length=400)[0]
        cut = 20 * SR + int(rms.argmin()) * 400
        pieces.append(audio[:cut]); audio = audio[cut:]
    return pieces + [audio]

chunks, meta = [], []
for split, fn in rows:
    path = f"{root}/{split}/{fn}" if fn.endswith(".wav") else f"{root}/{split}/{fn}.wav"
    audio, _ = librosa.load(path, sr=SR, mono=True)
    rms = librosa.feature.rms(y=audio)[0]
    meta.append(dict(split=split, filename=fn, duration=len(audio) / SR, speech_ratio=float((rms > 0.02).mean())))
    for p in split30(audio):
        chunks.append((len(meta) - 1, p))
print("clips", len(meta), "chunks", len(chunks), flush=True)

proc = WhisperProcessor.from_pretrained(MODEL)
model = WhisperForConditionalGeneration.from_pretrained(MODEL, dtype=torch.float16, attn_implementation="sdpa").cuda().eval()
prompt_ids = proc.get_prompt_ids(PROMPT, return_tensors="pt").cuda()

def bad(t):
    """Hallucination check: a phrase repeated 4x in a row, a prompt copy, or Whisper's compression-ratio heuristic."""
    w = t.lower().split()
    loop = any(w[i:i + n] == w[i + n:i + 2 * n] == w[i + 2 * n:i + 3 * n] == w[i + 3 * n:i + 4 * n]
               for n in range(2, 7) for i in range(max(len(w) - 4 * n + 1, 0)))
    return loop or len(t) / max(len(zlib.compress(t.encode())), 1) > 2.4 or "i mean, uh, you know, hmm" in t.lower()

def decode(pieces, prompt):
    feats = proc(pieces, sampling_rate=SR, return_tensors="pt").input_features.cuda().half()
    kw = dict(prompt_ids=prompt_ids) if prompt else {}
    with torch.no_grad():
        ids = model.generate(feats, language="en", task="transcribe", do_sample=False, num_beams=1, max_new_tokens=200, **kw)
    out = [t.strip() for t in proc.batch_decode(ids, skip_special_tokens=True)]
    return [t[len(PROMPT):].strip() if t.startswith(PROMPT) else t for t in out]  # some versions echo the prompt

piece_text, t0, redone = [""] * len(chunks), time.time(), 0
for b in range(0, len(chunks), BS):
    batch = chunks[b:b + BS]
    out = decode([p for _, p in batch], prompt=True)
    retry = [i for i, t in enumerate(out) if bad(t)]
    if retry:  # flagged pieces are decoded again without the prompt
        for i, t in zip(retry, decode([batch[i][1] for i in retry], prompt=False)):
            out[i] = t
        redone += len(retry)
    piece_text[b:b + len(out)] = out
    if b % (BS * 10) == 0:
        print(b, f"{time.time()-t0:.0f}s redone={redone} |", out[0][:150], flush=True)

texts = [""] * len(meta)
for (ci, _), t in zip(chunks, piece_text):
    texts[ci] = (texts[ci] + " " + t).strip()
df = pd.DataFrame(meta); df["transcript"] = texts; df["n_words"] = [len(t.split()) for t in texts]
df["flagged"] = df.transcript.map(bad)
df.to_csv("/kaggle/working/transcripts.csv", index=False)
print("done", len(df), f"{time.time()-t0:.0f}s | pieces re-decoded without prompt:", redone, "of", len(chunks),
      "| clips still flagged:", int(df.flagged.sum()))
print(df.sample(min(5, len(df)), random_state=0)[["filename", "transcript"]].to_string())
