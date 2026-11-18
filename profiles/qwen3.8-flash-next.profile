# qwen.rs profile, written by --tune-profile on 2026-11-13
# model Qwen3.8-Flash-Next (Flash(8)), context 262144, 24 logical CPUs, windows
# 96 tokens per trial, 2 rounds, depths [4290, 16384, 65536, 261024], 47 min
#
# Measured (decode, agentic coding conversation at each depth):
#   depth 4290 nothink: 18.76 tok/s (1.72 tok/step) tuned, 18.10 with the defaults; prefill 185 tok/s
#   depth 4290 think: 22.51 tok/s (2.32 tok/step) tuned, 22.38 with the defaults; prefill 185 tok/s
#   depth 16093 nothink: 26.88 tok/s (3.73 tok/step) tuned, 25.18 with the defaults; prefill 189 tok/s
#   depth 16093 think: 19.92 tok/s (1.85 tok/step) tuned, 19.78 with the defaults; prefill 189 tok/s
#   depth 65103 nothink: 28.04 tok/s (4.20 tok/step) tuned, 25.99 with the defaults; prefill 177 tok/s
#   depth 65103 think: 18.07 tok/s (1.79 tok/step) tuned, 17.95 with the defaults; prefill 177 tok/s
#   depth 259674 nothink: 20.68 tok/s (4.22 tok/step) tuned, 20.20 with the defaults; prefill 152 tok/s
#   depth 259674 think: 16.69 tok/s (1.91 tok/step) tuned, 15.91 with the defaults; prefill 152 tok/s
#
# Sampling is the model's (the cards' sets unless --profile gave others);
# the QWEN_* knobs are this machine's: rerun --tune-profile on another one.
# Format and precedence: crates/server/src/profile.rs.

[all]
QWEN_FLASH_MTP_COST = 67,86,104,118,141,152,187,201

[nothink]
temperature = 0.7
top_p = 0.8
top_k = 20
min_p = 0.0
presence_penalty = 1.5
frequency_penalty = 0.0
repetition_penalty = 1.0
QWEN_FLASH_MTP_N = 3
QWEN_FLASH_MTP_PMIN = 0.2
QWEN_SUFFIX_ENGINE_MIN = 6
QWEN_FLASH_MTP_PSCALE = 0.7

[think]
# Coupled drafting (the head's draw with the sampler's Gumbel noise; the
# sample stays an exact draw from the model) and the acceptance fitted for
# it at this mode's sampling (2026-11-13; the tuner keeps neither line: add
# them back by hand). Live Zed tool loop, step-priced tok/s over 4 / 2
# seeds: 12k 19.58 -> 19.89, 55k 17.96 -> 18.44; stand-in thinking set
# 21.0 -> 22.0 tok/s (3 rounds). Uncoupled fit: -0.934,-0.644.
QWEN_FLASH_COUPLED = 1
QWEN_FLASH_MTP_CAL = -0.583,-0.540
temperature = 1.0
top_p = 0.95
top_k = 20
min_p = 0.0
presence_penalty = 0.0
frequency_penalty = 0.0
repetition_penalty = 1.0
QWEN_FLASH_MTP_N = 3
QWEN_FLASH_MTP_PSCALE = 0.8
QWEN_FLASH_MTP_PMIN = 0.2
QWEN_SUFFIX_ENGINE_MIN = 6

[@8192]
QWEN_FLASH_MTP_COST = 67,83,106,120,142,156,186,198
# step costs measured at run time (added by hand: +1.6-2.5 % on the Zed
# tool loop at 12k, -0.4-0.9 % at 55k, so off from 32768)
QWEN_FLASH_MTP_AUTO = 1

[nothink@8192]
QWEN_SUFFIX_ENGINE_MIN = 4

[@32768]
QWEN_FLASH_MTP_COST = 70,87,109,120,148,164,195,205
QWEN_FLASH_MTP_AUTO = 0

[nothink@32768]
QWEN_FLASH_MTP_PSCALE = 0.8

[@130048]
QWEN_FLASH_MTP_COST = 81,105,133,156,180,206,241,254

[nothink@130048]
QWEN_SUFFIX_ENGINE_MIN = 6

[think@130048]
QWEN_SUFFIX_ENGINE_MIN = 12
