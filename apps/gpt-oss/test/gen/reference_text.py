# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "openai-harmony==0.0.8",
#   "tokenizers==0.23.1",
#   "huggingface_hub==1.32.0",
# ]
# ///
"""Record gpt-oss's text references from openai-harmony, and write the
tokenizer they need.

    uv run --python 3.12 apps/gpt-oss/test/gen/reference_text.py

Writes test/fixtures/harmony.json: the ids openai-harmony renders for a
prompt, as main renders it, and the messages its parser reads in a set of
completions. Writes test/checkpoints/tokenizer.json: gpt-oss-20b's
tokenizer at REVISION, trimmed by brot's trim_parity_tokenizers.py to what
those texts reach, and checks that it encodes and decodes them as the full
file does.
"""
import json, os, sys
from importlib.metadata import version
from huggingface_hub import hf_hub_download
from openai_harmony import (
    Conversation,
    HarmonyEncodingName,
    Message,
    ReasoningEffort,
    Role,
    StreamableParser,
    SystemContent,
    load_harmony_encoding,
)
from tokenizers import Tokenizer

REPO = "openai/gpt-oss-20b"
REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"
here = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(here, "..", "..", "..", "..", "packages", "brot", "test", "gen"))
import trim_parity_tokenizers as trim

# What main renders: one user turn, at medium reasoning effort.
PROMPTS = [
    {"name": "one user turn", "text": "What is 2 + 2?"},
    {"name": "user text that spells a special token", "text": "What does <|end|> mean?"},
]

COMPLETIONS = [
    {"name": "analysis then final",
     "text": "<|channel|>analysis<|message|>The user greets.<|end|>"
             "<|start|>assistant<|channel|>final<|message|>Hello!<|return|>"},
    {"name": "final only",
     "text": "<|channel|>final<|message|>4<|return|>"},
    {"name": "cut short",
     "text": "<|channel|>analysis<|message|>Let me think about"},
    {"name": "characters split across tokens",
     "text": "<|channel|>final<|message|>Hi 👋 there 👨‍👩‍👧‍👦, 𠜎𠜱 and "
             "สวัสดี ✨<|return|>"},
    {"name": "three channels",
     "text": "<|channel|>analysis<|message|>Plan.<|end|>"
             "<|start|>assistant<|channel|>commentary<|message|>Working.<|end|>"
             "<|start|>assistant<|channel|>final<|message|>Done.\n\nBye.<|return|>"},
    {"name": "a tool call stops the turn",
     "text": "<|channel|>analysis<|message|>Need the weather.<|end|>"
             "<|start|>assistant to=functions.get_weather<|channel|>commentary "
             '<|constrain|>json<|message|>{"city": "Paris"}<|call|>'},
]



def trimmed(fixture):
    """The full tokenizer's file cut to what [fixture]'s texts reach."""
    full = hf_hub_download(REPO, "tokenizer.json", revision=REVISION)
    tok = Tokenizer.from_file(full)
    with open(full, encoding="utf-8") as f:
        data = json.load(f)
    added = [t["content"] for t in data["added_tokens"]]
    ids = {t["id"] for t in data["added_tokens"]}
    seen = set()
    for c in fixture["prompts"]:
        ids |= set(c["ids"])
        seen |= trim.words(tok, c["text"], [])
    for c in fixture["completions"]:
        ids |= set(c["ids"])
        seen |= trim.words(tok, tok.decode(c["ids"], skip_special_tokens=False), added)
    seen |= trim.words(tok, enc_text(fixture), added)
    model = data["model"]
    trim.trim_bpe(model, {tok.id_to_token(i) for i in ids}, seen)
    for t in data["added_tokens"]:
        model["vocab"].setdefault(t["content"], t["id"])
    return tok, trim.dump(data)


def enc_text(fixture):
    return "".join(c["rendered"] for c in fixture["prompts"])


def main():
    enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    system = SystemContent.new().with_reasoning_effort(ReasoningEffort.MEDIUM)
    prompts = []
    for c in PROMPTS:
        messages = [
            Message.from_role_and_content(Role.SYSTEM, system),
            Message.from_role_and_content(Role.USER, c["text"]),
        ]
        ids = enc.render_conversation_for_completion(
            Conversation.from_messages(messages), Role.ASSISTANT
        )
        prompts.append({**c, "ids": ids, "rendered": enc.decode(ids)})

    stops = set(enc.stop_tokens_for_assistant_actions())
    completions = []
    for c in COMPLETIONS:
        ids = enc.encode(c["text"], allowed_special="all")
        parser = StreamableParser(enc, role=Role.ASSISTANT)
        for i in ids:
            parser.process(i)
        messages = [
            {"channel": m.channel, "content": m.content[0].text}
            for m in parser.messages
        ]
        if parser.current_content:
            messages.append(
                {"channel": parser.current_channel, "content": parser.current_content}
            )
        completions.append(
            {"name": c["name"], "ids": ids, "messages": messages,
             "stopped": ids[-1] in stops}
        )

    out = {
        "versions": {"openai-harmony": version("openai-harmony")},
        "tokenizer": {"repo": REPO, "revision": REVISION},
        "prompts": prompts,
        "completions": completions,
    }
    full, text = trimmed(out)
    path = os.path.join(here, "..", "checkpoints", "tokenizer.json")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    small = Tokenizer.from_file(path)
    for c in prompts:
        assert small.encode(c["text"], add_special_tokens=False).ids == full.encode(
            c["text"], add_special_tokens=False).ids, c["name"]
    for c in completions:
        assert small.decode(c["ids"], skip_special_tokens=False) == full.decode(
            c["ids"], skip_special_tokens=False), c["name"]
    with open(os.path.join(here, "..", "fixtures", "harmony.json"), "w") as f:
        json.dump(out, f, ensure_ascii=True)
        f.write("\n")
    print("wrote", path, os.path.getsize(path), "bytes")


main()
