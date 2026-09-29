# Stage A: verbatim ASR for every train+test clip -> transcripts.csv (shared by all experiments)
import glob, os, time
import pandas as pd, numpy as np, torch, librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration

MODEL = "openai/whisper-large-v3"  # MIT licence (CrisperWhisper v1/v2 are non-commercial -> break the OSI rule)
# Disfluent, error-laden prompt nudges Whisper to keep fillers, repeats and grammar errors instead of cleaning them up.
PROMPT = "Umm, so, uh, I I go to the the office yesterday and, hmm, he don't know nothing about it, like, you know."
SR, BS = 16000, 16

root = os.path.dirname(glob.glob("/kaggle/input/**/train.csv", recursive=True)[0])
rows = [("train", f) for f in pd.read_csv(f"{root}/train.csv").filename] + \
       [("test", f) for f in pd.read_csv(f"{root}/test.csv").filename]

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
model = WhisperForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.float16, attn_implementation="sdpa").cuda().eval()
prompt_ids = proc.get_prompt_ids(PROMPT, return_tensors="pt").cuda()

texts, t0 = [""] * len(meta), time.time()
for b in range(0, len(chunks), BS):
    batch = chunks[b:b + BS]
    feats = proc([p for _, p in batch], sampling_rate=SR, return_tensors="pt").input_features.cuda().half()
    with torch.no_grad():
        ids = model.generate(feats, language="en", task="transcribe", prompt_ids=prompt_ids,
                             do_sample=False, num_beams=1, max_new_tokens=200)
    for (ci, _), t in zip(batch, proc.batch_decode(ids, skip_special_tokens=True)):
        t = t.strip()
        if t.startswith(PROMPT):  # some versions keep the prompt in the decoded text
            t = t[len(PROMPT):].strip()
        texts[ci] = (texts[ci] + " " + t).strip()
    if b % (BS * 10) == 0:
        print(b, f"{time.time()-t0:.0f}s |", texts[batch[0][0]][:150], flush=True)

df = pd.DataFrame(meta); df["transcript"] = texts; df["n_words"] = [len(t.split()) for t in texts]
df.to_csv("/kaggle/working/transcripts.csv", index=False)
print("done", len(df), f"{time.time()-t0:.0f}s")
print(df.sample(5, random_state=0)[["filename", "transcript"]].to_string())
