# qwen.rs profile, written by --tune-profile on 2026-10-25
# model Qwen3.8-27B (GgufMtp(7)), context 262144, 24 logical CPUs, windows
# 128 tokens per trial, 3 rounds, depths [4290, 16384, 65536] at QWEN_VRAM_RESERVE_MB 1024 (86 min), then 260000 alone at 1536 (203 min)
#
# Measured (decode, agentic coding conversation at each depth):
#   depth 4290 nothink: 10.30 tok/s (3.37 tok/step) tuned, 10.24 with the defaults; prefill 115 tok/s
#   depth 4290 think: 8.83 tok/s (3.17 tok/step) tuned, 8.22 with the defaults; prefill 115 tok/s
#   depth 16125 nothink: 10.51 tok/s (4.24 tok/step) tuned, 10.41 with the defaults; prefill 101 tok/s
#   depth 16125 think: 7.83 tok/s (3.10 tok/step) tuned, 7.27 with the defaults; prefill 101 tok/s
#   depth 65124 nothink: 7.36 tok/s (4.59 tok/step) tuned, 7.27 with the defaults; prefill 62 tok/s
#   depth 65124 think: 5.67 tok/s (3.13 tok/step) tuned, 5.57 with the defaults; prefill 62 tok/s
#   depth 258788 nothink: 3.05 tok/s (3.82 tok/step) tuned, 2.94 with the defaults; prefill 32 tok/s
#   depth 258788 think: ~2.2-2.4 tok/s (2.69 tok/step), no knob past the margin
#
# Load with a bare --mtp (it loads this file's largest QWEN_MTP_N: 7 drafts).
# Sampling is the model's (the cards' sets unless --profile gave others);
# the QWEN_* knobs are this machine's: rerun --tune-profile on another one.
# Format and precedence: crates/server/src/profile.rs.

[nothink]
temperature = 0.7
top_p = 0.8
top_k = 20
min_p = 0.0
presence_penalty = 1.5
frequency_penalty = 0.0
repetition_penalty = 1.0
QWEN_MTP_N = 7
QWEN_MTP_PMIN = 0.8
QWEN_SUFFIX_MIN = 12

[think]
temperature = 1.0
top_p = 0.95
top_k = 20
min_p = 0.0
presence_penalty = 0.0
frequency_penalty = 0.0
repetition_penalty = 1.0
QWEN_MTP_N = 7
QWEN_SUFFIX_MIN = 12
QWEN_MTP_PMIN = 0.5

[think@32768]
QWEN_MTP_PMIN = 0.65

[nothink@131072]
QWEN_MTP_PMIN = 0.65

[think@131072]
QWEN_MTP_PMIN = 0.8
