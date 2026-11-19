# /// script
# requires-python = "==3.12.*"
# dependencies = [
#   "openai-whisper @ git+https://github.com/openai/whisper.git@86098128c0b4f24f0e2aa2994de830614b474227",
#   "torch==2.15.0",
#   "jiwer==4.0.0",
# ]
#
# [tool.uv]
# # Only whisper's text normaliser is used, which imports torch.
# override-dependencies = ["triton; sys_platform == 'never'"]
# sources = { torch = { index = "pytorch-cpu" } }
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Whether the rows transcribe the same text: each transcript of the
recording that raven.sh and the reference scripts wrote, as word error rates
after Whisper's English normaliser (whisper/normalizers at 86098128), against
LibriSpeech's reference and against openai-whisper's transcript by the same
model on the same device at float32.

    uv run apps/whisper/bench/wer.py
"""
import glob, os, re
import jiwer
from whisper.normalizers import EnglishTextNormalizer

DIR = os.path.join(
    os.environ.get("RAVEN_CACHE_ROOT")
    or os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "raven"),
    "whisper")
normalize = EnglishTextNormalizer()


def read(path):
    with open(path) as f:
        return normalize(f.read())


reference = read(os.path.join(DIR, "recording.txt"))
print(f"{'transcript':<44} {'WER vs LibriSpeech':>19} {'WER vs openai-whisper':>22}")
for path in sorted(glob.glob(os.path.join(DIR, "*-*-*.txt"))):
    name = os.path.basename(path)[:-4]
    m = re.match(r"(raven|openai-whisper|faster-whisper|hf-transformers|whisper-cpp)-(.+)-(cuda|cpu)(-fp16)?$", name)
    if not m:
        continue
    upstream = os.path.join(DIR, f"openai-whisper-{m[2]}-{m[3]}.txt")
    text = read(path)
    against = f"{100 * jiwer.wer(read(upstream), text):.2f}%" if os.path.exists(upstream) else "-"
    print(f"{name:<44} {100 * jiwer.wer(reference, text):18.2f}% {against:>22}")
