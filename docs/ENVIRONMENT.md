# Environment

The results in the thesis were produced on **two machines with two different
software stacks**. This is not incidental — the quantized system-under-test
variants only run under OpenVINO on Intel hardware, and that stack pins an
older `transformers` than the Apple Silicon one. Installing both into a single
environment will not resolve.

| | Apple Silicon (MPS) | Intel Arc workstation |
|---|---|---|
| Backend | `sut.backend: torch` | `sut.backend: openvino` |
| Python | 3.13.5 | 3.12.12 |
| torch | 2.8.0 | 2.10.0 |
| transformers | 5.3.0 | 4.55.4 |
| Extra | — | openvino 2025.4.1, optimum-intel 1.27.0, nncf 3.1.0 |
| SUTs | `Qwen/Qwen3.5-4B`, `Qwen/Qwen3.5-9B` | `OpenVINO/llava-v1.6-mistral-7b-hf-int8-ov`, `OpenVINO/Qwen2.5-VL-7B-Instruct-int4-ov` |
| Used for | most evolutionary and PDQ campaigns | Exp-23, Exp-26, Exp-104 Phase B (LLaVA) |

Which machine a given campaign ran on is recorded per experiment in
[REPRODUCTION.md](REPRODUCTION.md). Some run directories encode it in the name
(`Exp-03-mac` / `Exp-03-workstation`).

## Base install (torch backend)

```bash
git clone --recurse-submodules https://github.com/KaiserRuben/Masterthesis.git
cd Masterthesis

conda create -n uni python=3.13
conda activate uni

# Install the torch build matching your accelerator FIRST.
#   Apple Silicon:  pip install torch==2.8.0 torchvision==0.23.0
#   CUDA 12.4:      pip install torch==2.8.0 torchvision==0.23.0 \
#                       --index-url https://download.pytorch.org/whl/cu124
pip install -r experiments/requirements.txt
```

`requirements.txt` installs the SMOO framework from `tools/smoo` in editable
mode, so the submodule must be checked out. If you cloned without
`--recurse-submodules`:

```bash
git submodule update --init tools/smoo
```

Verify:

```bash
python -c "import smoo; print(smoo.__file__)"
pytest tests/          # expect 725 passed
```

## LaTeX (figure emitters only)

The thesis figure emitters typeset each figure to check its box dimensions, so
they need `pdflatex` on `PATH` — a system dependency pip will not report. It is
needed only to rebuild figures; nothing else in the repository uses it. They
also need the thesis tree, which is not part of this repository; see
[REPRODUCTION.md](REPRODUCTION.md).

## OpenVINO install (quantized SUTs)

Create a **separate** environment — see the version conflict above.

```bash
conda create -n uni-ov python=3.12
conda activate uni-ov
pip install -r experiments/requirements-openvino.txt
```

`requirements-openvino.txt` is standalone — it carries the whole dependency set
at the versions that environment ran. Do **not** install `requirements.txt` on
top of it (or it on top of `requirements.txt`): they pin different majors of
`transformers` and different torch builds, so either overlay downgrades the
other.

## Why the SMOO submodule points at a fork

`tools/smoo` tracks
[`KaiserRuben/SMOO`](https://github.com/KaiserRuben/SMOO) branch `masterarbeit`
rather than [`oliverweissl/SMOO`](https://github.com/oliverweissl/SMOO)
directly. The branch is three commits on top of upstream `6f17ccf`, carrying
the packaging and dependency-compatibility changes these experiments run
against:

1. **Packaging.** A setuptools `pyproject.toml` mapping `src/` onto the `smoo`
   package namespace, so `pip install -e tools/smoo` works — which is what
   `experiments/requirements.txt` does on every install. The StyleGAN
   internals' non-relative import fallbacks are rewritten to match, so they
   resolve from an installed distribution as well as a source checkout.
2. **timm 1.0 compatibility.** timm 1.0 moved `timm.models.layers.*` to
   `timm.layers.*` and made several `timm.models.X` modules private. The
   StyleGAN-XL checkpoints were pickled before that move and still name the old
   paths, so loading them against the pinned `timm==1.0.24` needs the legacy
   names resolved to their modern equivalents.
3. **Inference-mode manipulation.** `manipulate`, `get_w` and `get_images` run
   under `torch.inference_mode`, so the returned tensor does not retain the
   synthesis graph — which otherwise exhausts memory on CPU at non-trivial
   batch sizes.

None of these are thesis-specific; they are general compatibility fixes against
a newer dependency set than upstream currently targets, and are written to be
upstreamable unchanged. Nothing else diverges — the search behaviour is
upstream's.

## External services and data

**Redis** is an optional inference cache. `src/sut/vlm_sut.py` degrades
gracefully to no caching when no server is reachable, so it is not required to
reproduce any result — it only affects wall time. Configure with
`sut.redis_url`.

**ImageNet** cannot be redistributed. `src/data/imagenet.py` streams from
`ILSVRC/imagenet-1k` on the HuggingFace Hub, which requires accepting the
dataset licence on your account. Set `HF_TOKEN` in the environment. The cache
directory defaults to `~/.cache/imagenet`; pass `fallbacks=[...]` to point at
external storage.

**Model weights** download from the HuggingFace Hub on first use. The VQGAN
codebook neighbour table (`f8_16384_full.npz`) is generated by the
preprocessing scripts under `experiments/preprocessing/`.

## Configuring paths

The entry points you are most likely to run — the runners, the analysis
package, `analysis/viz/thesis/pgf/**` and `tools/render_manipulation_*.py` —
resolve their own location and take the rest from the environment:

| Variable | Meaning | Default |
|---|---|---|
| `THESIS_DIR` | Thesis checkout the figure scripts emit into | `analysis/outputs/thesis` inside the repo |
| `ANALYSIS_ASSET_ROOT` | Where `analysis/core/style.py` writes figures | the author's Obsidian vault if present, else `analysis/outputs/assets` |
| `HF_TOKEN` | HuggingFace access for ImageNet and model weights | — |

Beyond those, **absolute paths from the original machines survive in roughly a
hundred tracked files** and you will have to edit them:

- **Configs.** Most files under `configs/` point `image.knn_path` and the
  ImageNet cache at `/mnt/storage/...` or `/Volumes/...`. This includes
  `configs/templates/`, so even the documented starting point needs editing.
- **Launch scripts.** `configs/Exp-104/launch_qwen_ab.sh` and the Exp-105
  drivers hardcode the repository root and `ssh` to a specific host.
- **Ad-hoc analysis scripts.** The `.py` files tracked alongside the aggregates
  under `experiments/analysis/output/**` mostly begin with a
  `sys.path.insert("/Users/...")` and absolute input/output constants. They are
  kept as a record of how each aggregate was produced, not as a supported
  entry point — the aggregates themselves are what the figures read.
- **Older figure scripts.** `analysis/viz/thesis/exp_*.py` and
  `render_exp104_*.py` predate the `THESIS_DIR` convention; the `pgf/`
  emitters listed in [REPRODUCTION.md](REPRODUCTION.md) are the current path.

## Known test flake

`tests/test_evolutionary.py::TestVLMBoundaryTester::test_trace_row_count` and
`tests/test_discrete_optimizer.py::TestConstruction::test_initial_population_shape`
fail intermittently because pymoo's `setup()` is not seeded in those fixtures.
Rerun before assuming a change caused it; both pass in isolation.
