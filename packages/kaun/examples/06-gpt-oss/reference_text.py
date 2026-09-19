# Reference values for the text side of kaun's gpt-oss example: what the
# o200k_harmony tokenizer makes of a set of strings, what the harmony renderer
# makes of a set of conversations, and what the harmony parser reads in a set of
# completions. Writes fixtures/harmony.json next to this script.
#
# Two references are consulted: the tiktoken encoding inside openai-harmony, and
# the tokenizer.json of the checkpoint repository read by HuggingFace
# tokenizers, which is the file the example loads. They must agree on every
# string but those that spell <|endofprompt|>, which tokenizer.json holds as
# token 200018 and the harmony encoding reads as plain text: tokenizer.json is
# recorded there.
#
#   uv run --with openai-harmony==0.0.8 --with tokenizers==0.23.2 \
#          --with huggingface_hub==1.32.0 reference_text.py
import hashlib, json, os
import openai_harmony, tokenizers, huggingface_hub
from importlib.metadata import version
from huggingface_hub import hf_hub_download
from openai_harmony import (
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    ReasoningEffort,
    Role,
    StreamableParser,
    SystemContent,
    load_harmony_encoding,
)

REPO = "tiny-random/gpt-oss-mxfp4"

SPECIALS = [
    "<|startoftext|>",
    "<|endoftext|>",
    "<|return|>",
    "<|constrain|>",
    "<|channel|>",
    "<|start|>",
    "<|end|>",
    "<|message|>",
    "<|call|>",
    "<|endofprompt|>",
]

TEXTS = [
    "",
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "I'm sure they'll say it's what he'd want; DON'T, I'VE said.",
    "HTTPServer parseXMLDocument camelCase snake_case SCREAMING_CASE",
    "def f(x):\n    return [i ** 2 for i in range(x) if i % 3 == 0]\n",
    "let rec fib n = if n < 2 then n else fib (n - 1) + fib (n - 2)\n",
    "\tif (a != b) {\n\t\treturn a->next;  // trailing\n\t}\n",
    "https://example.com/a/b?c=d&e=f#g //comment /* block */",
    "1 22 333 4444 55555 3.14159 1,000,000 0x1F 1e-9",
    "Où est passé l'été à Zürich, señor Åström? Čeština, Tiếng Việt.",
    "é ä ñ precomposed éäñ",
    "日本語のテキストと中文文本、그리고 한국어.",
    "Ελληνικά, кириллица, עברית, العربية, हिन्दी, ไทย",
    "👋 🌍 👨‍👩‍👧‍👦 🇫🇷 👍🏽 ✨",
    "a" + " " * 40 + "b",
    "  leading and trailing  ",
    "line\n\n\nbreaks\r\nand\ttabs \n \n  x",
    "\n\n\n\n",
    " ",
    " non-breaking em space​zero width",
    "� replacement",
    "x" * 300,
    "math: ∑_{i=0}^{n} x_i² ≤ ∞ ⇒ ∀ε>0",
    "".join(SPECIALS),
    " ".join(SPECIALS),
    "<|start|>assistant<|channel|>analysis<|message|>Think.<|end|>"
    "<|start|>assistant<|channel|>final<|message|>Done.<|return|>",
    "<|start|>assistant to=functions.f<|channel|>commentary "
    '<|constrain|>json<|message|>{"a": 1}<|call|>',
    "<|reserved_200000|><|reserved_200017|>",
    "not special: <|nope|> <|start |> < |end|> <|START|>",
]

DATE = "2026-09-19"


def user(text):
    return {"role": "user", "content": text}


def assistant(channel, text):
    return {"role": "assistant", "channel": channel, "content": text}


CONVERSATIONS = [
    {"name": "system only", "date": None, "effort": "medium",
     "instructions": None, "messages": []},
    {"name": "one user turn", "date": None, "effort": "medium",
     "instructions": None, "messages": [user("What is 2 + 2?")]},
    {"name": "dated", "date": DATE, "effort": "medium",
     "instructions": None, "messages": [user("What day is it?")]},
    {"name": "low effort", "date": DATE, "effort": "low",
     "instructions": None, "messages": [user("Hi")]},
    {"name": "high effort", "date": DATE, "effort": "high",
     "instructions": None, "messages": [user("Prove it.")]},
    {"name": "developer instructions", "date": DATE, "effort": "medium",
     "instructions": "Answer in French.\nBe brief.",
     "messages": [user("Où est la gare ?")]},
    {"name": "user text that spells a special token", "date": None,
     "effort": "medium", "instructions": None,
     "messages": [user("What does <|end|> mean?")]},
    {"name": "multi-turn", "date": DATE, "effort": "low",
     "instructions": "Be terse.",
     "messages": [
         user("Name a prime."),
         assistant("analysis", "Any prime will do."),
         assistant("final", "7"),
         user("Another, 日本語で."),
     ]},
    {"name": "analysis kept after the first final", "date": None,
     "effort": "medium", "instructions": None,
     "messages": [
         user("a"),
         assistant("analysis", "one"),
         assistant("commentary", "aside"),
         assistant("final", "A"),
         user("b"),
         assistant("analysis", "two"),
         assistant("final", "B"),
         user("c"),
     ]},
    {"name": "analysis kept while no final closes the turn", "date": None,
     "effort": "medium", "instructions": None,
     "messages": [
         user("a"),
         assistant("analysis", "one"),
         assistant("final", "A"),
         user("b"),
         assistant("analysis", "two"),
     ]},
]

# What follows <|start|>assistant. The third has no stop token; the fourth
# splits an emoji and a CJK character across tokens.
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

EFFORTS = {
    "low": ReasoningEffort.LOW,
    "medium": ReasoningEffort.MEDIUM,
    "high": ReasoningEffort.HIGH,
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    path = hf_hub_download(REPO, "tokenizer.json")
    hf = tokenizers.Tokenizer.from_file(path)

    strings = []
    for text in TEXTS:
        ids = hf.encode(text, add_special_tokens=False).ids
        if "<|endofprompt|>" not in text:
            assert ids == enc.encode(text, allowed_special="all"), text
            assert enc.decode(ids) == text, text
        assert hf.decode(ids, skip_special_tokens=False) == text, text
        strings.append({"text": text, "ids": ids})

    conversations = []
    for c in CONVERSATIONS:
        system = SystemContent.new().with_reasoning_effort(EFFORTS[c["effort"]])
        if c["date"] is not None:
            system = system.with_conversation_start_date(c["date"])
        messages = [Message.from_role_and_content(Role.SYSTEM, system)]
        if c["instructions"] is not None:
            developer = DeveloperContent.new().with_instructions(c["instructions"])
            messages.append(Message.from_role_and_content(Role.DEVELOPER, developer))
        for m in c["messages"]:
            if m["role"] == "user":
                messages.append(Message.from_role_and_content(Role.USER, m["content"]))
            else:
                messages.append(
                    Message.from_role_and_content(
                        Role.ASSISTANT, m["content"]
                    ).with_channel(m["channel"])
                )
        ids = enc.render_conversation_for_completion(
            Conversation.from_messages(messages), Role.ASSISTANT
        )
        conversations.append({**c, "ids": ids, "text": enc.decode(ids)})

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
        "versions": {
            "openai-harmony": version("openai-harmony"),
            "tokenizers": tokenizers.__version__,
        },
        "tokenizer": {"repo": REPO, "sha256": sha256(path)},
        "strings": strings,
        "conversations": conversations,
        "completions": completions,
    }
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "fixtures", "harmony.json"), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=True)
        f.write("\n")


main()
