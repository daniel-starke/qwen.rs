"""Turn the per-layer decode cuts at S positions (`cut_all.py --s S` on the
unrolled stateless IR) into speculative-verify graphs: every delta-net layer
additionally outputs the recurrent state after each position
(`qwen_rs.unroll.state{i}` -> results `state{i}`, f16) and the whole conv
window (the concat of the past window and the S new inputs, `conv_full`)
instead of the window after the last position, so the host can roll back to
any accepted prefix without recomputing. Attention layers and the head are
copied unchanged (their KV rollback is a host-side window choice).

    python verify_cut.py SRC_DIR OUT_DIR [--layers 64]

Also writes OUT_DIR/outs.txt (each layer's result names in order) for
`npubench`."""
import os, sys, shutil
import openvino as ov
import openvino.opset13 as op

src, out = sys.argv[1], sys.argv[2]
NL = int(sys.argv[sys.argv.index("--layers") + 1]) if "--layers" in sys.argv else 64
os.makedirs(out, exist_ok=True)
core = ov.Core()
lines = []
for L in range(NL):
    m = core.read_model(f"{src}/layer{L}.xml")
    ops = {o.get_friendly_name(): o for o in m.get_ordered_ops()}
    states = sorted([o for n, o in ops.items() if ".unroll.state" in n], key=lambda o: int(o.get_friendly_name().rsplit("state", 1)[1]))
    if not states:  # attention layer: unchanged
        for ext in ("xml", "bin"):
            shutil.copyfile(f"{src}/layer{L}.{ext}", f"{out}/layer{L}.{ext}")
        names = []
        for r in m.get_results():
            ns = r.input_value(0).get_names()
            names.append(sorted(ns)[0] if ns else "?")
        lines.append(f"{L} " + " ".join(names))
        continue
    results = []
    keep = []
    for r in m.get_results():
        nm = sorted(r.input_value(0).get_names())[0]
        if nm.startswith("cache_params.present.ssm"):
            continue  # = the last state{i}
        if nm.startswith("cache_params.present.conv"):
            # Result <- Convert <- Slice(window) <- ... take the Slice's input
            node = r.input_value(0).get_node()
            while node.get_type_name() in ("Convert",):
                node = node.input_value(0).get_node()
            assert node.get_type_name() == "Slice", node.get_type_name()
            full = node.input_value(0)
            if full.get_element_type() != ov.Type.f16:
                full = op.convert(full, "f16").output(0)
            rr = op.result(full)
            rr.output(0).get_tensor().set_names({"conv_full"})
            keep.append((rr, "conv_full"))
            continue
        keep.append((r, nm))
    for i, s in enumerate(states):
        rr = op.result(op.convert(s.output(0), "f16"))
        rr.output(0).get_tensor().set_names({f"state{i}"})
        keep.append((rr, f"state{i}"))
    mm = ov.Model([r for r, _ in keep], m.get_parameters(), f"verify{L}")
    mm.validate_nodes_and_infer_types()
    ov.save_model(mm, f"{out}/layer{L}.xml", compress_to_fp16=False)
    lines.append(f"{L} " + " ".join(n for _, n in keep))
    print(f"layer{L}: {len(states)} states, conv_full {list(full.get_partial_shape())}", flush=True)
for ext in ("xml", "bin"):
    shutil.copyfile(f"{src}/head.{ext}", f"{out}/head.{ext}")
open(f"{out}/outs.txt", "w").write("\n".join(lines) + "\n")
print("->", out)
