# /// script
# requires-python = "==3.13.*"
# dependencies = [
#   "soundfile==0.14.0",
# ]
# ///
"""A fixed subset of LibriSpeech's test-clean and test-other
(https://www.openslr.org/12), the utterances greedy.py decodes: for each
speaker of each set, the first 3 utterances in (speaker, chapter,
utterance) numeric order.

    uv run apps/whisper/test/gen/subset.py LIBRISPEECH_DIR SUBSET_TSV

LIBRISPEECH_DIR holds SPEAKERS.TXT, test-clean/ and test-other/. Writes
SUBSET_TSV, a row per utterance: set, id, speaker, chapter, sex, path under
LIBRISPEECH_DIR, the FLAC file's sha256, samples, duration and reference
transcript.
"""
import hashlib, os, sys

import soundfile as sf

root, out_tsv = sys.argv[1:3]
SETS = ["test-clean", "test-other"]
PER_SPEAKER = 3


def speakers_sex():
    sex = {}
    for line in open(os.path.join(root, "SPEAKERS.TXT"), encoding="utf-8"):
        if line.startswith(";"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 3:
            sex[parts[0]] = parts[1]
    return sex


def utterances(s):
    out = []
    base = os.path.join(root, s)
    for spk in os.listdir(base):
        for ch in os.listdir(os.path.join(base, spk)):
            d = os.path.join(base, spk, ch)
            for line in open(os.path.join(d, f"{spk}-{ch}.trans.txt"), encoding="utf-8"):
                uid, text = line.rstrip("\n").split(" ", 1)
                key = tuple(int(x) for x in uid.split("-"))
                out.append((key, uid, os.path.join(s, spk, ch, uid + ".flac"), text))
    out.sort()
    return out


def main():
    sex = speakers_sex()
    rows, total = [], {}
    for s in SETS:
        utts = utterances(s)
        total[s] = (len(utts), len({k[0] for k, *_ in utts}))
        taken = {}
        for key, uid, rel, text in utts:
            if taken.get(key[0], 0) >= PER_SPEAKER:
                continue
            taken[key[0]] = taken.get(key[0], 0) + 1
            path = os.path.join(root, rel)
            info = sf.info(path)
            assert (info.samplerate, info.channels, info.subtype) == (16000, 1, "PCM_16"), (path, info)
            h = hashlib.sha256(open(path, "rb").read()).hexdigest()
            rows.append({
                "set": s, "id": uid, "speaker": str(key[0]), "chapter": str(key[1]),
                "sex": sex.get(str(key[0]), "?"), "path": rel, "sha256": h,
                "samples": str(info.frames), "duration_s": f"{info.frames / 16000:.4f}",
                "transcript": text,
            })
    cols = list(rows[0])
    with open(out_tsv, "w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(r[c] for c in cols) + "\n")

    for s in SETS:
        sub = [r for r in rows if r["set"] == s]
        dur = sum(float(r["duration_s"]) for r in sub)
        words = sum(len(r["transcript"].split()) for r in sub)
        print(f"{s}: {total[s][0]} utterances, {total[s][1]} speakers in the set; subset {len(sub)} "
              f"utterances, {len({r['speaker'] for r in sub})} speakers, {dur:.1f} s, {words} words, "
              f"max {max(float(r['duration_s']) for r in sub):.2f} s")


main()
