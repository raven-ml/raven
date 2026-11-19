# /// script
# requires-python = "==3.12.*"
# dependencies = ["numpy==2.3.5", "soundfile==0.13.1"]
# ///
"""The files bench_whisper.ml and the reference scripts read, into
$RAVEN_CACHE_ROOT/whisper (by default ~/.cache/raven/whisper):

    uv run apps/whisper/bench/fetch.py

- recording.wav: chapter 122797 of LibriSpeech's test-clean (speaker 672,
  8 min 16 s, CC BY 4.0, https://www.openslr.org/12), its utterances in
  order as one 16 kHz 16-bit file; clip.wav, its first minute; and
  recording.txt, their transcripts;
- the released checkpoints tiny, small, large-v3-turbo and large-v3 (4.9 GB),
  each checked against the SHA-256 its URL names;
- ranks/gpt2.tiktoken and ranks/multilingual.tiktoken, upstream's rank files
  at 86098128.

LibriSpeech's archive (346 MB) is read and left out.
"""
import hashlib, io, os, tarfile, urllib.request
import numpy as np
import soundfile

DIR = os.path.join(
    os.environ.get("RAVEN_CACHE_ROOT")
    or os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "raven"),
    "whisper")
LIBRISPEECH = "https://www.openslr.org/resources/12/test-clean.tar.gz"
LIBRISPEECH_SHA256 = "39fde525e59672dc6d1551919b1478f724438a95aa55f874b576be21967e6c23"
CHAPTER = "LibriSpeech/test-clean/672/122797/"
MODELS = "https://openaipublic.azureedge.net/main/whisper/models"
CHECKPOINTS = {
    "tiny": "65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9",
    "small": "9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794",
    "large-v3-turbo": "aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a",
    "large-v3": "e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb",
}
RANKS = "https://raw.githubusercontent.com/openai/whisper/86098128c0b4f24f0e2aa2994de830614b474227/whisper/assets"


def fetch(url, path, sha256=None):
    if os.path.exists(path):
        return
    h = hashlib.sha256()
    with urllib.request.urlopen(url) as r, open(path + ".part", "wb") as f:
        while chunk := r.read(1 << 20):
            h.update(chunk)
            f.write(chunk)
    if sha256 and h.hexdigest() != sha256:
        raise SystemExit(f"{url}: SHA-256 mismatch")
    os.replace(path + ".part", path)


def recording():
    if os.path.exists(os.path.join(DIR, "recording.wav")):
        return
    data = urllib.request.urlopen(LIBRISPEECH).read()
    if hashlib.sha256(data).hexdigest() != LIBRISPEECH_SHA256:
        raise SystemExit("LibriSpeech: SHA-256 mismatch")
    flacs, lines = {}, []
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        for m in tar:
            if m.name.startswith(CHAPTER) and m.name.endswith(".flac"):
                flacs[os.path.basename(m.name)[:-5]] = tar.extractfile(m).read()
            elif m.name.startswith(CHAPTER) and m.name.endswith(".trans.txt"):
                lines = tar.extractfile(m).read().decode().splitlines()
    ids = [line.split(" ", 1)[0] for line in lines]
    samples = [soundfile.read(io.BytesIO(flacs[i]), dtype="int16") for i in ids]
    assert all(rate == 16000 for _, rate in samples)
    x = np.concatenate([x for x, _ in samples])
    soundfile.write(os.path.join(DIR, "clip.wav"), x[:60 * 16000], 16000, subtype="PCM_16")
    soundfile.write(os.path.join(DIR, "recording.wav"), x, 16000, subtype="PCM_16")
    with open(os.path.join(DIR, "recording.txt"), "w") as f:
        f.write(" ".join(line.split(" ", 1)[1] for line in lines) + "\n")


os.makedirs(os.path.join(DIR, "ranks"), exist_ok=True)
recording()
for name, sha256 in CHECKPOINTS.items():
    fetch(f"{MODELS}/{sha256}/{name}.pt", os.path.join(DIR, name + ".pt"), sha256)
for name in ["gpt2", "multilingual"]:
    fetch(f"{RANKS}/{name}.tiktoken", os.path.join(DIR, "ranks", name + ".tiktoken"))
