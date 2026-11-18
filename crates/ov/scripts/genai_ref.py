"""Reference run of the export through OpenVINO GenAI: python genai_ref.py DEVICE [max_new]."""
import os
import sys, time
import openvino_genai as g

dev = sys.argv[1]
max_new = int(sys.argv[2]) if len(sys.argv) > 2 else 32
path = os.environ.get("OV_EXPORT") or sys.exit("OV_EXPORT: the OpenVINO export")
t = time.time()
cfg = {}
if dev == "NPU":
    cfg = {"MAX_PROMPT_LEN": 1024, "MIN_RESPONSE_LEN": 256}
pipe = g.VLMPipeline(path, dev, **cfg)
print(f"load {time.time() - t:.1f} s", flush=True)
gc = g.GenerationConfig()
gc.max_new_tokens = max_new
gc.do_sample = False
for prompt in ["The capital of France is", "Write a haiku about rust."]:
    t = time.time()
    r = pipe.generate(prompt, generation_config=gc)
    print(repr(r.texts[0]), flush=True)
    m = r.perf_metrics
    print(f"  ttft {m.get_ttft().mean:.0f} ms  tpot {m.get_tpot().mean:.1f} ms  ({m.get_throughput().mean:.2f} tok/s)  total {time.time() - t:.1f} s", flush=True)
