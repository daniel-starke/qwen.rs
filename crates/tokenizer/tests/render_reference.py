"""Regenerate tests/data/tools_prompt.txt from the model's real chat template.

"""
"""Run from the repo root with a Python that has Jinja2:

    python crates/tokenizer/tests/render_reference.py \\
        crates/tokenizer/tests/data/tools_prompt.txt

data/chat_template.jinja is `tokenizer.chat_template` copied verbatim out of
Qwen3.8-27B-Q4_K_M.gguf (Qwen3.8-Flash-Next's chat_template.jinja is the same
file, byte for byte). Regenerate only when that template changes; the
checked-in output is what `tools_prompt_matches_the_models_chat_template`
asserts against. The `think_*_prompt.txt` files beside it (thinking on, at
each reasoning effort, with and without preserve_thinking) are written too,
for `thinking_prompts_match_the_template`."""
import io, json, sys
from jinja2 import Environment
from jinja2.exceptions import TemplateError

import os
HERE = os.path.dirname(os.path.abspath(__file__))
TPL = io.open(os.path.join(HERE, "data", "chat_template.jinja"), encoding="utf-8").read()


def raise_exception(msg):
    raise TemplateError(msg)


def tojson(x, **kw):
    # transformers renders tool JSON with ensure_ascii=False and no HTML escaping.
    return json.dumps(x, ensure_ascii=False)


env = Environment(trim_blocks=False, lstrip_blocks=False)
env.filters["tojson"] = tojson
env.globals["raise_exception"] = raise_exception
tpl = env.from_string(TPL)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "Read",
            "description": "Read a file from disk.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Absolute path."},
                    "limit": {"type": "integer", "description": "Max lines."},
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Bash",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
]

MESSAGES = [
    {"role": "system", "content": "You are a terse assistant."},
    {"role": "user", "content": "How many lines are in src/main.rs?"},
    {
        "role": "assistant",
        "content": "I will look.",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "Read",
                    "arguments": {"file_path": "/repo/src/main.rs", "limit": 100},
                },
            }
        ],
    },
    {"role": "tool", "content": "fn main() {}\n"},
    {"role": "user", "content": "And now run the tests."},
]

def write(path, out):
    io.open(path, "w", encoding="utf-8", newline="").write(out)
    print(path, repr(out[-120:]))
    print("bytes", len(out.encode("utf-8")))


write(
    sys.argv[1],
    tpl.render(
        messages=MESSAGES,
        tools=TOOLS,
        add_generation_prompt=True,
        enable_thinking=False,
    ),
)

# Thinking on: past reasoning replayed, an effort's instruction in the system
# turn. Written beside the first file; `thinking_prompts_match_the_template`
# asserts against them.
THINK_MESSAGES = [
    {"role": "system", "content": "You are a terse assistant."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4.", "reasoning_content": "Simple arithmetic."},
    {"role": "user", "content": "How many lines are in src/main.rs?"},
    {
        "role": "assistant",
        "content": "I will look.",
        "reasoning_content": "\nRead the file first.\n",
        "tool_calls": [
            {
                "type": "function",
                "function": {"name": "Read", "arguments": {"file_path": "/repo/src/main.rs"}},
            }
        ],
    },
    {"role": "tool", "content": "fn main() {}\n"},
]
DIR = os.path.dirname(os.path.abspath(sys.argv[1]))
write(
    os.path.join(DIR, "think_xhigh_prompt.txt"),
    tpl.render(messages=THINK_MESSAGES, tools=TOOLS, add_generation_prompt=True),
)
write(
    os.path.join(DIR, "think_low_nopreserve_prompt.txt"),
    tpl.render(
        messages=THINK_MESSAGES,
        add_generation_prompt=True,
        reasoning_effort="low",
        preserve_thinking=False,
    ),
)
write(
    os.path.join(DIR, "think_medium_nosystem_prompt.txt"),
    tpl.render(
        messages=THINK_MESSAGES[1:2],
        add_generation_prompt=True,
        reasoning_effort="medium",
    ),
)
write(
    os.path.join(DIR, "think_xhigh_nosystem_prompt.txt"),
    tpl.render(messages=THINK_MESSAGES[1:2], add_generation_prompt=True),
)
