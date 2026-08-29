# Environment

The thesis results were produced on **two machines with two software stacks**.
This is not incidental: the quantized system-under-test variants only run under
OpenVINO on Intel hardware, and that stack pins an older `transformers` than
the Apple Silicon one. They do not co-install.

| | Apple Silicon (MPS) | Intel Arc workstation |
|---|---|---|
| Backend | `sut.backend: torch` | `sut.backend: openvino` |
| Python | 3.13.5 | 3.12.12 |
| torch | 2.8.0 | 2.10.0 |
| transformers | 5.3.0 | 4.55.4 |
| Extra | — | openvino 2025.4.1, optimum-intel 1.27.0, nncf 3.1.0 |
| SUTs | `Qwen/Qwen3.5-4B`, `Qwen/Qwen3.5-9B` | `OpenVINO/llava-v1.6-mistral-7b-hf-int8-ov`, `OpenVINO/Qwen2.5-VL-7B-Instruct-int4-ov` |
| Used for | most evolutionary and PDQ campaigns | Exp-23, Exp-26, Exp-104 Phase B (LLaVA) |
| Install | [below](#base-install-torch-backend) | [below](#openvino-install-quantized-suts) |

Which machine ran a given campaign is recorded per experiment in
[REPRODUCTION.md](REPRODUCTION.md). Some run directories encode it in the name
(`Exp-03-mac` / `Exp-03-workstation`).

## Base install (torch backend)

```bash
git clone --recurse-submodules https://github.com/KaiserRuben/Masterthesis.git
cd Masterthesis

conda create -n uni python=3.13
conda activate uni

# Install the torch build matching your accelerator FIRST — requirements.txt
# pins torch without a platform tag, so pip resolves the wrong wheel otherwise.
#   Apple Silicon:  pip install torch==2.8.0 torchvision==0.23.0
#   CUDA 12.4:      pip install torch==2.8.0 torchvision==0.23.0 \
#                       --index-url https://download.pytorch.org/whl/cu124
pip install -r experiments/requirements.txt
```

`requirements.txt` installs SMOO from `tools/smoo` in editable mode, so the
submodule must be checked out. If you cloned without `--recurse-submodules`:

```bash
git submodule update --init tools/smoo
```

Verify:

```bash
python -c "import smoo; print(smoo.__file__)"
pytest tests/          # 725 tests
```

Pins are exact because several results are sensitive to tokenizer and
model-loading behaviour that changed across transformers 4.x/5.x.

## OpenVINO install (quantized SUTs)

A **separate** environment — see the version conflict above.

```bash
conda create -n uni-ov python=3.12
conda activate uni-ov
pip install -r experiments/requirements-openvino.txt
```

`requirements-openvino.txt` is standalone: it carries the whole dependency set
at the versions that environment ran. Do **not** layer it and
`requirements.txt` in either order — they pin different majors of
`transformers` and different torch builds, so either overlay downgrades the
other.

## LaTeX (figure emitters only)

The thesis figure emitters typeset each figure to measure its box, so they need
`pdflatex` on `PATH` — a system dependency pip will not report. Nothing else in
the repository uses it. They also need the thesis tree, which is not part of
this repository; see [REPRODUCTION.md](REPRODUCTION.md#what-it-takes-to-rebuild-a-figure).

## Configuring paths

Three environment variables, all optional:

| Variable | Meaning | Default |
|---|---|---|
| `HF_TOKEN` | HuggingFace access for ImageNet and model weights | — (required to run a search) |
| `THESIS_DIR` | Thesis checkout the figure emitters write into | `analysis/outputs/thesis` — writable, but not a working build |
| `ANALYSIS_ASSET_ROOT` | Where `analysis/core/style.py` writes PNGs | the author's Obsidian vault if present, else `analysis/outputs/assets` |

The entry points you are most likely to run — the runners, the analysis
package, `analysis/viz/thesis/pgf/**` and `tools/render_manipulation_*.py` —
resolve their own location and take the rest from the environment.

### Absolute paths in configs

`configs/templates/` and `configs/Exp-10/` are portable: they use
`~/.cache/imagenet` as the writable primary and list an external drive as a
fallback, which `src/data/imagenet.py` skips when it is not mounted. Start from
those.

Most other configs are not. Of the 299 config files:

| Key | Value | Files |
|---|---|---|
| `image.knn_cache_path` | `/mnt/storage/huggingface/vqgan_knn/f8_16384_full.npz` | 210 |
| `image.knn_cache_path` | `~/.cache/vqgan_knn/f8_16384_full.npz` | 50 |
| `cache_dirs[0]` (writable primary) | `/mnt/storage/huggingface/imagenet` | 210 |

`/mnt/storage/huggingface/` is the original workstation. Rewrite those two keys
before reusing one of those configs — a fallback entry is skipped when absent,
but the *primary* cache dir is where the code writes.

Three further categories still carry machine-specific absolutes and are kept as
a record rather than as supported entry points:

- **Launch scripts.** `configs/Exp-104/launch_qwen_ab.sh` and the Exp-105
  drivers hardcode the repository root and `ssh` to a specific host.
- **Ad-hoc analysis scripts.** The `.py` files tracked alongside the aggregates
  under `experiments/analysis/output/**` mostly open with
  `sys.path.insert("/Users/...")` and absolute I/O constants. They document how
  each aggregate was produced; the aggregates themselves are what the figures
  read.
- **Older figure scripts.** `analysis/viz/thesis/exp_*.py` and
  `render_exp104_*.py` predate the `THESIS_DIR` convention. The `pgf/` emitters
  in [REPRODUCTION.md](REPRODUCTION.md#figure-index) are the current path.

## External services and data

**ImageNet** cannot be redistributed. `src/data/imagenet.py` streams
`ILSVRC/imagenet-1k` from the HuggingFace Hub, which requires accepting the
dataset licence on your account; export `HF_TOKEN`. The cache defaults to
`~/.cache/imagenet` — the first entry of `cache_dirs` is the writable primary,
the rest are read-only fallbacks skipped when absent.

**Model weights** download from the Hub on first use. The VQGAN codebook
neighbour table (`f8_16384_full.npz`) is generated by the scripts under
`experiments/preprocessing/`.

**Redis** is an optional inference cache. `src/sut/vlm_sut.py` degrades
gracefully to no caching when no server is reachable, so it affects wall time
only, never a result. Configure with `sut.redis_url`.

## Why the SMOO submodule points at a fork

`tools/smoo` tracks
[`KaiserRuben/SMOO`](https://github.com/KaiserRuben/SMOO) branch `masterarbeit`
rather than [`oliverweissl/SMOO`](https://github.com/oliverweissl/SMOO)
directly — three commits on top of upstream `6f17ccf`:

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

None are thesis-specific; they are general compatibility fixes against a newer
dependency set than upstream currently targets, written to be upstreamable
unchanged. The search behaviour is upstream's.

## Known test flake

`tests/test_evolutionary.py::TestVLMBoundaryTester::test_trace_row_count` and
`tests/test_discrete_optimizer.py::TestConstruction::test_initial_population_shape`
fail intermittently because pymoo's `setup()` is unseeded in those fixtures.
Both pass in isolation — rerun before assuming a change caused it.
