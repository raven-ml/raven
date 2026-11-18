# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "openai-harmony==0.0.8",
# ]
# ///
"""Record gpt-oss's text references from openai-harmony: what its renderer
makes of a set of conversations, and what its parser reads in a set of
completions.

    uv run apps/gpt-oss/test/gen/reference_text.py

Writes test/fixtures/harmony.json, which names the checkpoint whose
tokenizer.json the app reads, at REVISION.
"""
import json, os
from importlib.metadata import version
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
REVISION = "27b6ad8040614834e65239f102de94bc459f48e5"

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


def main():
    enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

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
        "versions": {"openai-harmony": version("openai-harmony")},
        "tokenizer": {"repo": REPO, "revision": REVISION},
        "conversations": conversations,
        "completions": completions,
    }
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "..", "fixtures", "harmony.json"), "w") as f:
        json.dump(out, f, ensure_ascii=True)
        f.write("\n")


main()
