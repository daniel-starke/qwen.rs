# HTTP server

`cli MODEL --serve [PORT] [--host H]` loads the model once and serves it over
HTTP/1.1, one thread per connection. The server crate has no third-party
dependencies. Routing, JSON and SSE are hand-written
(`crates/server/src/lib.rs`).

There is no authentication, rate limiting or request-size policy, so bind
the server to localhost.

## Endpoints

| Route | Description |
|---|---|
| `POST /v1/messages` | Anthropic Messages API with `model`, `messages`, `system`, `max_tokens`, `temperature`, `top_p`, `top_k`, `stream`, `stop_sequences`, `tools`, `tool_choice`, `thinking`. Streamed as `message_start`, `content_block_start`, `content_block_delta` (`text_delta` / `input_json_delta`), `content_block_stop`, `message_delta`, `message_stop`. |
| `POST /v1/messages/count_tokens` | `{"input_tokens": N}` for the prompt the request would render, tools included. Image blocks are not encoded for the count, so images undercount. |
| `POST /v1/chat/completions` (also `/chat/completions`) | OpenAI Chat Completions as llama-server serves it, streamed or whole, with `tools` / `tool_calls` / `tool` messages, `image_url` parts (`data:` URLs), `stream_options.include_usage` (with `cached_tokens`), `stop`. Without `max_tokens` it generates until the context is full. |
| `GET /v1/models` (also `/models`) | The model id plus llama-server's `meta` (`n_ctx`, `n_ctx_train`, …). The id is `Qwen3.8-27B`, or `Qwen3.8-Flash-Next` on an `arch = flash` engine. `/props` and a response to a request that names no model carry the same id. |
| `GET /props` | `default_generation_settings.n_ctx`, `modalities.vision`, `chat_template_caps`. |
| `GET /health` | `{"status":"ok"}` |
| `POST /mcp` | MCP (Streamable HTTP) with the image tools `generate_image` / `edit_image`, with `--image-model` or an image model as the first argument. See [image.md](image.md). |
| `POST /v1/images/generations`, `POST /v1/images/edits` | OpenAI Images API, under the same conditions as `/mcp`. |

Both APIs translate into one internal request and share the prompt
rendering, the tool format and the prefix cache, so a request through one
API can reuse the cached state of the other.

A prompt that leaves no room for one token in the context is refused (400).
Generation stops at the end of the context.

**Images.** A `content` array may carry Anthropic `image` blocks (base64) or
`data:image/...;base64,...` URLs, in a message or inside a tool result. That covers a
`tool_result`'s content, or an OpenAI `tool` message's `image_url` part (an
agent's `read_file` on a picture). An image inside a tool result goes inside
the `<tool_response>`, as the chat template renders it. On a GGUF the server
needs `--mmproj`, and an NPU engine needs `vision` in its `.npu` file.
Without them an image is a 400, not silently dropped. Remote `http(s)` URLs
are refused, because the server does not fetch.

## Tool use

The API side is JSON (`tools`, `tool_use`, `tool_result`). The model side is
its chat template, where **a call is XML-style text, not JSON**:

```
<tool_call>
<function=Bash>
<parameter=command>
ls /tmp
</parameter>
</function>
</tool_call>
```

Tool definitions go into the system turn as OpenAI-shaped
`{"type":"function","function":{…}}` JSON. Tool results come back as user
turns wrapped in `<tool_response>`, and past assistant turns carry an empty
`<think>` block. The renderer reproduces the model's own Jinja template byte
for byte. `crates/tokenizer/tests/data/chat_template.jinja` is
`tokenizer.chat_template` copied verbatim from the GGUF, and a test compares
the output against Jinja2's.

- **Argument types come from the schema.** The text format carries no types,
  so `<parameter=n>123</parameter>` is a string for a parameter declared
  `string` and a number for one declared `integer`. Undeclared parameters are
  parsed as JSON, and kept as text when that fails.
- **`tool_choice` is enforced structurally.** `any` seeds the assistant turn
  with `<tool_call>`, and `tool` also seeds `<function=NAME>`, so the call is
  the only continuation the format allows. `none` drops the tool preamble.
- **Special tokens follow the GGUF's token types.** `<tool_call>`, `<think>`
  and similar are USER_DEFINED tokens. BPE must not split them, and they are
  not hidden from the output. Only CONTROL tokens are hidden.
  `cli MODEL --tokenize TEXT` shows the ids.
- **No repetition penalty on the server**, because the call format repeats
  tokens by construction.
- Tools declared by `type` without an `input_schema` (server-side tools such
  as `web_search`) are skipped.

`QWEN_SERVER_DEBUG=1` prints the exact prompt and the raw completion of every
request. Look at it first when a tool call does not come back.

## Thinking

- **Off by default on `/v1/messages`.** `"thinking": {"type": "enabled" |
  "adaptive", "budget_tokens": N}` turns it on (the budget is enforced),
  `output_config.effort` (`low` / `medium` / `high` / `max`) sets the effort,
  and `thinking` blocks are streamed and replayed.
- **On by default on `/v1/chat/completions`**, as in llama-server, unless the
  request says otherwise (`chat_template_kwargs.enable_thinking`,
  `reasoning_effort`). Zed's llama.cpp provider sends no switch.
- `--serve --think LEVEL` sets the default of both APIs, and `--think off` turns
  it off everywhere.

## Sampling

Sampling is seeded per request. A non-standard `seed` field reproduces a run.

Requests that leave `temperature`, `top_p` or `top_k` out get the profile's
values for their thinking mode and prompt length. The built-in values are
the model cards' sets. Thinking uses 1.0 / 0.95 / 20, and non-thinking uses 0.7 / 0.8 / 20
with presence penalty 1.5. `temperature: 0` gets no penalties it does not
ask for. `min_p`, `presence_penalty`, `frequency_penalty` and
`repetition_penalty` are accepted too and apply to the generated tokens
only. See "Profiles" in the README.

**Client field overrides.** `--client-ignore PREFIX=FIELD[,FIELD...]`
(repeatable) drops those top-level request fields from clients whose
`User-Agent` starts with `PREFIX` (`*` = every client), so the profile's
values apply instead. The default is `*=temperature` because clients send fixed
values (Zed sends `temperature: 1.0` on every request, also with thinking
off), so every client's temperature is replaced by the profile's for the
request's mode, and a request cannot ask for greedy decoding. Giving the
flag replaces the default, and `--client-ignore off` disables it.

**Coupled sampling.** On a drafting NPU engine the profile can turn on
coupled sampling (`QWEN_FLASH_COUPLED`, on for Flash-Next's thinking
requests). It has the same distribution, drawn with noise fixed by seed and
position, which the MTP head's drafts share.

## Prefix cache

One cache serves every engine (`crates/prefixcache`). A request restores the
longest kept state that is a proper prefix of its prompt and prefills only
the rest. The state stays in the cache, so an edited message or a retried
turn restores from it again. Each request logs `prefix cache: reused N of M
prompt tokens`.

- **Checkpoints** are taken at the prompt's last turn, at shared prefixes
  (the tool preamble, a system turn of 1024 or more tokens), where a prompt
  branches off a cached one, and every `QWEN_CACHE_INTERVAL` tokens (default
  4096), so an edit inside a long prompt replays at most to the next
  checkpoint.
- **Pages.** The KV is kept in pages of 256 positions, and a page equal to
  the corresponding page of another state is stored once. A conversation's
  checkpoints therefore cost one recurrent state each (GGUF ~150 MB, NPU 27B
  ~75 MB, Flash-Next ~113 MB) plus the KV once. A state is charged for the
  positions it holds, not for `--ctx`.
- **Budget and eviction.** `--prompt-cache-mb` sets the budget (default 32768, `0`
  disables it). Eviction is by cost, not LRU. The state whose loss costs the
  least prefill time per freed byte goes first (weighted by recency, hits
  and sharing).
- **On disk.** `--prompt-cache-dir DIR` (budget `--prompt-cache-disk-mb`,
  default the RAM budget) writes the shared prefixes, so a restarted server
  skips them. With `--prompt-cache-spill` every state evicted from RAM goes
  to disk too and is read back when that is faster than the prefill. Writes
  are limited for flash wear. A write must save enough prefill per GB,
  within a daily budget. A restored state is bit-exact.
- **Images** are part of the key (a hash of their pixels replaces the
  placeholder ids), so an image inside a reused prefix is not encoded again.
- **Concurrent requests** on one prefix both restore it. A request that would
  extend a state while another request is still storing it waits for that
  request.

Both NPU engines (the 27B and Flash-Next) use the same cache (`snapshot` /
`restore` of their state), together with speculative decoding from their MTP
block and context drafts, also from earlier responses. The OpenVINO export
backend has none of this. It resets and prefills every request. Two more
mechanisms apply to the NPU engines:

- **Live continuation** (`QWEN_ENGINE_LIVE`, default on). The engine keeps
  its state after a request, and the server remembers which tokens it holds
  (the prompt plus the reply). A request whose prompt extends them continues
  that state directly, with no reset, no restore and no second prefill of
  the reply, so an agent's next turn prefills only the tool result. Anything
  else (another conversation, an edited message, images, a failure) goes
  through the prefix cache.
- **Piece-aware checkpoints.** An engine that prefills in fixed pieces
  (Flash-Next uses 4096 tokens, each with a fixed cost) drops an interval
  checkpoint that would split a follow-up's feed into one more piece.

## Clients

**Zed**, either provider:

- Anthropic-compatible (`language_models.anthropic_compatible`, settings in
  the README) supports everything, including the thinking toggle and the effort
  selector (the model entry needs `"mode": {"type": "adaptive"}`). The model
  name is `"Qwen3.8"` on every engine (the server does not check the
  request's `model`). Changing the effort moves an instruction at the start
  of the system turn, so the first request after a change misses the prefix
  cache.
- llama.cpp uses the API URL `http://localhost:8080`. Zed reads the context window
  from `/props`, so start the server with `--ctx`. There is no thinking
  toggle, so the server's default applies.
- For image tools, use `"context_servers": {"qwen-image": {"url":
  "http://localhost:8080/mcp"}}` with `"context_server_timeout": 600` (Zed's
  default of 60 s cancels image jobs, and 600 is its maximum). Zed declares
  no MCP roots, so start the server with `--image-out-dir DIR` (a directory
  that holds the projects). Otherwise every `output_path` is refused.

`/v1/models` carries no `status` field on purpose. That field tells Zed it is
talking to a multi-model router, which would make Zed poll routes this
server does not have.

**Claude Code** posts to `/v1/messages?beta=true`, and the query is ignored. It
rejects unknown model ids before sending, so set `ANTHROPIC_MODEL` to an id
it knows (the server ignores the field). See the README for the full set of
variables. Its first turn is a ~30k-token prompt, and with `--prompt-cache-dir`
that prefill is paid once per machine.

## Concurrency

On a GGUF, requests run concurrently and each has its own run state. The
NPU engines and the OpenVINO backend run one request at a time, and a second
request waits. An NPU-engine server holds the machine-wide NPU lock for its
whole life, so exit it when done.

An image job holds every device. Chat requests wait while it runs, and it
waits for running chat requests.

Every SSE response closes its connection (`Connection: close`), so a client
never reuses a streamed connection for another request.

## Measuring a server

- `--progress` (or `QWEN_PROGRESS=1`) draws a progress bar on stderr for each
  request's prefill.
- `QWEN_NPU_TIMING=1` (NPU engines) prints per request a timeline (body,
  parse and render, lock wait, prefill, first token, and decode split into
  host, drafts, accept and verify), a drafts line (verified and accepted per
  source) and the `[eff]` / `[pool]` lines (CPU time, energy, time per call
  site).
- `--record-requests DIR` saves every POST request.
  `cargo run --release -p server --example replay -- DIR [HOST:PORT]` posts
  them again and prints each one's time to first content and total time.
