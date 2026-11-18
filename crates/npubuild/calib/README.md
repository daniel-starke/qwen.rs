# Calibration texts (frozen)

The NPU builds (`cli BF16_DIR --npu-build OUT`) calibrate their per-layer
Hessians, static ranges and GPTQ on these texts when no `--npu-calib FILE` is
given. The 27B reads `base.txt` and Flash-Next all three, with one 256-token chunk
from each in turn (`--npu-calib-chunks N` chunks in all).

**Never edit, reformat or regenerate these files.** Every byte is part of the
calibration. A change moves every weight a build derives from it and needs a
new accuracy measurement. They are copies, not links, so that changes to the
code or the docs they once came from do not reach them. `.gitattributes`
keeps git from converting their line endings. A deliberate recalibration
adds a new file (and a measurement) instead of editing one.

| file | content | source |
|---|---|---|
| `base.txt` | Rust source (context drafts) and project prose | this project, frozen copy |
| `code.txt` | Rust source of the server's tool use and OpenAI routes | this project, frozen copy |
| `poetry.txt` | 30 poems in English, German and French | public domain, below |

`poetry.txt` holds only works in the public domain worldwide (every author
died before 1915), taken from Project Gutenberg transcriptions with the
Gutenberg header, license and trademark, editors' notes, footnote marks and
line numbers removed. The authors are Shakespeare (Sonnets XVIII, XXIX, LXXIII, CXVI, CXXX),
Blake, Goethe (*Faust*, "Zueignung"), Keats, Baudelaire, Dickinson, Poe
("The Raven"), Burns, Trakl ("Die junge Magd"), Whitman, Wordsworth
("Tintern Abbey"), Shelley, Tennyson ("The Lady of Shalott", Part I) and
Coleridge (*The Rime of the Ancyent Marinere*, Part I, 1798 text).
