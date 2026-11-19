# /// script
# requires-python = "==3.13.*"
# dependencies = [
#   "openai-whisper @ git+https://github.com/openai/whisper.git@86098128c0b4f24f0e2aa2994de830614b474227",
#   "torch==2.15.0",
#   "numpy==2.5.5",
#   "tiktoken==0.14.0",
#   "jiwer==4.0.0",
#   # Beyond the direct pins: whisper's normaliser runs regex and
#   # more-itertools, and jiwer aligns with rapidfuzz.
#   "regex==2026.11.18",
#   "more-itertools==11.2.1",
#   "rapidfuzz==3.14.6",
# ]
#
# [tool.uv]
# # whisper declares triton on Linux x86-64 for its CUDA kernels; the
# # generators run on the CPU.
# override-dependencies = ["triton; sys_platform == 'never'"]
# sources = { torch = { index = "pytorch-cpu" } }
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Normaliser and WER goldens for test_english and test_wer: upstream's
EnglishTextNormalizer (whisper/normalizers/english.py at 86098128) on
LibriSpeech's references and upstream's greedy hypotheses and on edge
strings, and jiwer's word counts (jiwer 4.0.0, process_words with
wer_default) per model, protocol and set as notebooks/LibriSpeech.ipynb
computes the WER, and on random word sequences that tie alignments.

    uv run apps/whisper/test/gen/normalize.py SUBSET_TSV GREEDY_DIR OUT

OUT is apps/whisper/test/upstream, SUBSET_TSV subset.py's subset of
LibriSpeech, and GREEDY_DIR holds greedy.py's MODEL-PROTOCOL.jsonl for tiny,
tiny.en and base.en and both protocols.
Writes OUT/normalize.json: {"strings": [[text, normalised]...],
  "corpora": [{model, protocol, set, refs: [i...], hyps: [i...], counts}...],
  "pairs": [{reference, hypothesis, counts}...]}
where i indexes "strings" and counts are hits, substitutions, deletions,
insertions and wer.
"""
import json, os, random, sys

import jiwer
from whisper.normalizers import EnglishTextNormalizer

subset_tsv, greedy_dir, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
norm = EnglishTextNormalizer()

EDGE = [
    "", " ", "  leading and trailing  ", "tabs\tand\nnewlines\r\n", "nbsp here",
    "em space", "ideographic　space", "Hello, World!", "HELLO WORLD.",
    "[MUSIC] hello <noise> there", "(laughs) okay (sighs) fine", "a (b) c [d] e <f> g",
    "unclosed [bracket and <angle", "nested [a [b] c] d", "empty () and (x)", "(()",
    "hmm mm mhm mmm uh um umm hmmm uh-huh", "Um, well, uh, I think so.",
    "I won't, can't, and let's not; ain't y'all gonna wanna? Gotta go.",
    "I'ma do it, imma do it; woulda coulda shoulda, ma'am.",
    "Mr. Smith, Mrs. Jones, Dr. Who, St. Louis, Prof. X, Capt. Hook, Gov. Y",
    "Ald. A Gen. B Sen. C Rep. D Pres. E Rev. F Hon. G Asst. H Assoc. I Lt. J Col. K Jr. Sr. Esq.",
    "He'd been there. She's been here. They'd gone. It's gone. We'd done it. He's got it.",
    "don't isn't they're it's he'd we'll can't've I'm you've o'clock rock 'n' roll",
    "Mr Smith mrs", "the mr. mrs. st", "STREET st. peter",
    "1,234,567 and 1,2,3 and 12,345.67", "3.14 and 2. and .5 and end.", "U.S.A. and e.g.",
    "one two three", "twenty one", "twenty-one", "one hundred and five", "two thousand and twenty",
    "nineteen eighty four", "one oh one", "zero zero seven", "double seven", "triple zero",
    "double you", "two point five", "point five", "three and a half", "and a half",
    "one and a half hours", "a half", "five percent", "five per cent", "per cent",
    "twenty dollars", "twenty dollars and seven cents", "$2 and ¢7", "$2 ¢7", "$0.05",
    "£5 and €10", "five pounds", "a pound of flesh", "minus five", "plus two", "negative",
    "positive thinking", "minus", "the 1960s", "the nineteen sixties", "1st 2nd 3rd 4th",
    "first second third fourth fifth twelfth twentieth", "twenty first", "one hundredth",
    "a million", "two million", "three billion dollars", "1.5 million", "one point five million",
    "a thousand and one nights", "twenty decillion", "nine hundred ninety nine decillion",
    "one", "ones", "1", "1s", "$1", "1%", "1.5", "3.1", "one two", "the one and only",
    "seven elevens", "sixes and sevens", "the sixties", "eighties", "zero", "oh",
    "i have 2 apples and 3 oranges", "abc123def", "123abc", "4 th", "5 s", "2nd hand",
    "café naïve résumé façade", "Œuvre œuf Ærø Straße ẞ đ Đ ð Ð þ Þ ł Ł ø",
    "ﬁne ﬂour ½ ² ³ ① Ⅻ", "１２３ ＡＢＣ", "١٢٣ and ٤,٥", "Ελληνικά ΣΟΦΟΣ", "İstanbul",
    "colour honour organise centre theatre travelled", "mhm mmm",
    "symbols: #hash @at &amp * + = ~ ^ | \\ / _ ` {brace}",
    "quotes “smart” ‘single’ and — dash – en",
    "it’s won’t", "50% off", "100 % sure", "a% b", "$ alone", "¢ alone", "£ £5",
    "price $ 5", "the . period", "...", "a..b", "1.2.3.4", "version 2.0", "3.",
    "point", "and", "double", "triple", "five and", "and five", "hundred", "a hundred",
    "thousands", "millions of", "hundreds and thousands", "ten thousandth",
    "two hundred thousand five hundred", "one thousand nine hundred eighty four",
    "five hundred and sixty seven thousand", "the 3rd of may", "4 5 6", "zero point zero one",
    "I paid $20 million", "twenty million dollars", "one hundred and one dalmatians",
    "he said \"hello\"", "'quoted' words", " 'tis", "dogs ' tails", "o'brien",
    "tom ' s", "rock ' n ' roll",
]

strings, index = [], {}


def add(text):
    if text not in index:
        index[text] = len(strings)
        strings.append([text, norm(text)])
    return index[text]


for s in EDGE:
    add(s)

refs = {}
with open(subset_tsv, encoding="utf-8") as f:
    header = f.readline().rstrip("\n").split("\t")
    for line in f:
        r = dict(zip(header, line.rstrip("\n").split("\t")))
        refs[r["id"]] = r["transcript"]

corpora = []
for model in ["tiny", "tiny.en", "base.en"]:
    for protocol in ["notimestamps", "timestamps"]:
        path = os.path.join(greedy_dir, f"{model}-{protocol}.jsonl")
        recs = [json.loads(l) for l in open(path, encoding="utf-8")]
        for s in ["test-clean", "test-other"]:
            rs = [r for r in recs if r["set"] == s]
            ri = [add(refs[r["id"]]) for r in rs]
            hi = [add(r["text"]) for r in rs]
            o = jiwer.process_words([strings[i][1] for i in ri], [strings[i][1] for i in hi])
            corpora.append({"model": model, "protocol": protocol, "set": s, "refs": ri, "hyps": hi,
                            "counts": [o.hits, o.substitutions, o.deletions, o.insertions],
                            "wer": o.wer})

# Pairs: random words from a small vocabulary, so that alignments tie, and
# jiwer's own transforms (runs of spaces, tabs, empty sides).
rng = random.Random(20261119)


def words(n, vocab):
    return " ".join(rng.choice(vocab) for _ in range(n))


pairs = [("", ""), ("a", ""), ("", "a b"), ("  a   b  ", "a b"), ("a\tb c", "a b c"),
         ("a \t b", "a b"), ("a b c", "a b c"), ("a b c d", "e f"), ("x", "y z w")]
for _ in range(150):
    pairs.append((words(rng.randint(0, 9), "abc"), words(rng.randint(0, 9), "abc")))
for _ in range(50):
    pairs.append((words(rng.randint(0, 30), "abcdefg"), words(rng.randint(0, 30), "abcdefg")))
pair_cases = []
for ref, hyp in pairs:
    o = jiwer.process_words(ref, hyp)
    pair_cases.append({"reference": ref, "hypothesis": hyp,
                       "counts": [o.hits, o.substitutions, o.deletions, o.insertions],
                       "wer": o.wer})

with open(os.path.join(out, "normalize.json"), "w", encoding="utf-8") as f:
    json.dump({"strings": strings, "corpora": corpora, "pairs": pair_cases}, f,
              ensure_ascii=False, separators=(",", ":"))
print("strings", len(strings), "corpora", len(corpora), "pairs", len(pair_cases),
      "bytes", os.path.getsize(os.path.join(out, "normalize.json")))
for c in corpora:
    print(c["model"], c["protocol"], c["set"], c["counts"], "%.4f" % c["wer"])
