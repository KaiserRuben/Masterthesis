r"""Thesis figure: exp-config-effects — three configuration knobs on one search.

Three A/B experiments that each hold the search fixed and vary a single part of
the configuration.  The quantity is always the same: the targeted balance
``TgtBal`` (nats), and always its floor — the smallest value the run ever
reached.  Panel (a) shows that floor accumulating over generations, panels
(b) and (c) show only the end of it.

(a) modality (Exp-24).  LLaVA-NeXT-7B INT8, hammerhead shark -> spotted
    salamander, 300 generations x population 30.  One run per modality: the
    optimizer may edit both channels (joint), the prompt only (text-only), or
    the image only (image-only).  Plotted is the running minimum of
    ``pop_min_TgtBal``, i.e. the best floor reached up to and including that
    generation, so each curve is monotone by construction.  Joint gets four
    decades, text-only two, image-only none: the image-only arm ends at 1.4907
    after 300 generations, 10% below where it started.

(b) backend (Exp-26).  LLaVA-NeXT-7B INT8, image-only searches, 100 x 20, three
    gap-filter seeds.  The image channel is the only thing that differs: nearest
    neighbours in the VQGAN codebook (VQGAN-KNN), the same codebook restricted
    to a 20 deg double cone around the origin->target segment (VQGAN-cone), or
    StyleGAN-XL latents.  The three seeds carry three different class pairs, so
    the bars are read within a seed, never across.  Seed 83 is the pair from
    panel (a): the two VQGAN arms sit at 1.38 where the panel-(a) image-only arm
    sat at 1.49, and StyleGAN-XL drops the same pair to 0.0066.

(c) cone (Exp-27).  Qwen3.5-4B on the roster-locked panel-(a) pair, 100 x 20,
    sweeping the one tuning knob of the cone filter, its half-angle alpha.  The
    cone-off run is the reference line.  No alpha beats it.

Colours never reuse the two SUT colours of the Exp-104 figures, and never the
project's blue/vermillion answer coding: nothing here distinguishes systems or
answers.  Panel (a) is black / purple / green for the three modality arms.
Panels (b) and (c) share one reading: grey = no cone, teal = cone, and (b) adds
plum for the run that swaps the generator instead of the candidate set.  The
three (b) fills also separate in print greyscale (relative luminance 0.34 /
0.22 / 0.08).

Data sources
    runs/Exp-24/exp24_llava_ov_{joint,text_only,image_only}_seed_83_*/
        convergence.parquet, column pop_min_TgtBal over generations 0..299.
        exp24_llava_ov_joint_seed_83_1778274555 is an empty shell (0 files)
        superseded by ..._1778280071.  exp24_qwen_mps_joint_seed_83_1778459253
        is a different SUT and never wrote a stats.json; both are asserted
        present and excluded.
    runs/Exp-26/exp26_llava_ov_{vqgan_baseline,vqgan_cone,stylegan}_seed_{1,2,83}_*/
        convergence.parquet, floor = min pop_min_TgtBal over 100 generations.
        Excluded and asserted: four aborted stylegan seed-1 directories with no
        files at all, the stale vqgan_baseline seed-1 run 1779393770 (superseded
        7.0 ks later by 1779400099, and reading 2.6e-6 instead of 2.5e-5), and
        the single Qwen run in the same campaign.
    runs/Exp-27/exp27_qwen_mps_pairA_{baseline,cone05,cone10,cone20,cone40}_seed_0_*/
        Same floor.  cone20's convergence.parquet is truncated (no parquet magic
        bytes), so its floor is taken as the minimum over its 271 archived
        pareto_*.json fitness vectors instead.  That is an upper bound in
        principle; on all four intact arms the same reduction reproduces the
        convergence floor to the last bit, which the script asserts.  The point
        is drawn hollow and labelled.
    configs/Exp-2{4,6,7}/*.yaml  — modality, budget and cone alpha_deg are read
        back from the configs rather than hard-coded.

Produces
    figures/results/exp-config-effects.pdf
    figures/results/exp-config-effects.png

Usage (from the Masterarbeit repo root, conda env `uni`):
    PYTHONPATH=$PWD conda run --no-capture-output -n uni \
        python analysis/viz/thesis/exp_config_effects.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from _common import FS_ANN, FS_LABEL, FS_TICK, W_FULL, rect, save, setup

REPO = Path("/Users/kaiser/Projects/Masterarbeit")
RUNS = REPO / "runs"
CONFIGS = REPO / "configs"

SLUG = "exp-config-effects"
PAIR = ["hammerhead shark", "spotted salamander"]
LLAVA = "OpenVINO/llava-v1.6-mistral-7b-hf-int8-ov"
QWEN = "Qwen/Qwen3.5-4B"
CROSS_AT = 1e-2          # crossing criterion used throughout the thesis

# --- panel (a): Exp-24 -----------------------------------------------------
MODES = ("joint", "text_only", "image_only")
MODE_LABEL = {"joint": "joint", "text_only": "text-only",
              "image_only": "image-only"}
SUPPORT_GENS = (0, 10, 50, 100, 299)
# Published support points (thesis text).  4 dp, so the tolerance is 5e-5.
SUPPORT = {
    "joint":      (0.4272, 0.1503, 0.0010, 0.0003, 0.0002),
    "text_only":  (0.6167, 0.1895, 0.0420, 0.0193, 0.0046),
    "image_only": (1.6600, 1.5549, 1.4914, 1.4914, 1.4907),
}
EXP24_EMPTY = "exp24_llava_ov_joint_seed_83_1778274555"   # 0 files
EXP24_QWEN = "exp24_qwen_mps_joint_seed_83_1778459253"    # other SUT, no stats

# --- panel (b): Exp-26 -----------------------------------------------------
BACKENDS = ("vqgan_baseline", "vqgan_cone", "stylegan")
BACKEND_LABEL = {"vqgan_baseline": "VQGAN-KNN", "vqgan_cone": "VQGAN-cone",
                 "stylegan": "StyleGAN-XL"}
BACKEND_CFG = {"vqgan_baseline": "llava_ov_vqgan_baseline",
               "vqgan_cone": "llava_ov_vqgan_cone",
               "stylegan": "llava_ov_stylegan"}
SEEDS = (1, 2, 83)
EXP26_RUN = {
    ("vqgan_baseline", 1): "exp26_llava_ov_vqgan_baseline_seed_1_1779400099",
    ("vqgan_baseline", 2): "exp26_llava_ov_vqgan_baseline_seed_2_1779402707",
    ("vqgan_baseline", 83): "exp26_llava_ov_vqgan_baseline_seed_83_1779405328",
    ("vqgan_cone", 1): "exp26_llava_ov_vqgan_cone_seed_1_1779413043",
    ("vqgan_cone", 2): "exp26_llava_ov_vqgan_cone_seed_2_1779415729",
    ("vqgan_cone", 83): "exp26_llava_ov_vqgan_cone_seed_83_1779418459",
    ("stylegan", 1): "exp26_llava_ov_stylegan_seed_1_1779466321",
    ("stylegan", 2): "exp26_llava_ov_stylegan_seed_2_1779503897",
    ("stylegan", 83): "exp26_llava_ov_stylegan_seed_83_1779542170",
}
EXP26_FLOOR = {                       # published, 6 dp
    ("vqgan_baseline", 1): 2.5272e-05, ("vqgan_cone", 1): 2.27451e-04,
    ("stylegan", 1): 5.27263e-04,
    ("vqgan_baseline", 2): 1.000094, ("vqgan_cone", 2): 0.999598,
    ("stylegan", 2): 0.278604,
    ("vqgan_baseline", 83): 1.380108, ("vqgan_cone", 83): 1.381205,
    ("stylegan", 83): 0.006580,
}
EXP26_ABORTED = ("exp26_llava_ov_stylegan_seed_1_1779460815",
                 "exp26_llava_ov_stylegan_seed_1_1779464127",
                 "exp26_llava_ov_stylegan_seed_1_1779464752",
                 "exp26_llava_ov_stylegan_seed_1_1779465857")
EXP26_STALE = "exp26_llava_ov_vqgan_baseline_seed_1_1779393770"
EXP26_QWEN = "exp26_qwen_mps_vqgan_baseline_seed_1_1779394828"

# --- panel (c): Exp-27 -----------------------------------------------------
ARMS = ("baseline", "cone05", "cone10", "cone20", "cone40")
ARM_ALPHA = {"baseline": None, "cone05": 5.0, "cone10": 10.0,
             "cone20": 20.0, "cone40": 40.0}
EXP27_RUN = {
    "baseline": "exp27_qwen_mps_pairA_baseline_seed_0_1779714722",
    "cone05": "exp27_qwen_mps_pairA_cone05_seed_0_1779717753",
    "cone10": "exp27_qwen_mps_pairA_cone10_seed_0_1779722277",
    "cone20": "exp27_qwen_mps_pairA_cone20_seed_0_1779726440",
    "cone40": "exp27_qwen_mps_pairA_cone40_seed_0_1779733211",
}
EXP27_FLOOR = {"baseline": 2.4263, "cone05": 2.5208, "cone10": 2.5904,
               "cone20": 2.6611, "cone40": 2.5406}
EXP27_BOUNDED = "cone20"      # parquet truncated; floor from pareto_*.json
EXP27_N_PARETO = 271          # archived fronts in the cone20 directory
EXP27_ABORTED = ("exp27_qwen_mps_pairA_stylegan_seed_0_1779742531",
                 "exp27_qwen_mps_pairA_stylegan_seed_0_1779746194",
                 "exp27_qwen_mps_pairA_stylegan_seed_0_1779749247",
                 "exp27_qwen_mps_pairA_stylegan_seed_0_1779750436")

# --- palette ---------------------------------------------------------------
C_JOINT, C_TEXT, C_IMAGE = "#1A1A1A", "#6A51A3", "#2E7D4F"
MODE_COLOR = {"joint": C_JOINT, "text_only": C_TEXT, "image_only": C_IMAGE}
MODE_DASH = {"joint": (0, ()), "text_only": (0, (3.4, 1.9)),
             "image_only": (0, (5.0, 1.6, 1.0, 1.6))}

C_NOCONE, C_CONE, C_SGAN = "#9E9E9E", "#2E8B8A", "#7B2D5E"
BACKEND_COLOR = {"vqgan_baseline": C_NOCONE, "vqgan_cone": C_CONE,
                 "stylegan": C_SGAN}
# The hatch on a crossing bar is drawn in the bar's edge colour, so it needs a
# different one per fill to stay visible.
BACKEND_HATCH_EC = {"vqgan_baseline": "#4A4A4A", "vqgan_cone": "white",
                    "stylegan": "white"}
C_REF = "#6E6E6E"        # cone-off reference line and marker in panel (c)
C_GRID = "0.85"

# --- inch budget -----------------------------------------------------------
W = W_FULL
L_A, GAP_AB, GAP_BC, R_PAD = 0.56, 0.60, 0.52, 0.10
AXW_A, AXW_B, AXW_C = 1.76, 1.75, 1.40
assert abs(L_A + AXW_A + GAP_AB + AXW_B + GAP_BC + AXW_C + R_PAD - W) < 1e-9
X_A = L_A
X_B = X_A + AXW_A + GAP_AB
X_C = X_B + AXW_B + GAP_BC

B_PAD, B_LABEL, B_TICKS = 0.05, 0.20, 0.15
BOTTOM = B_PAD + B_LABEL + B_TICKS
T_HEAD, T_PAD = 0.22, 0.03
AXH = 2.20
H = BOTTOM + AXH + T_HEAD + T_PAD


# ---------------------------------------------------------------------------
# loading + provenance
# ---------------------------------------------------------------------------
def _floor(run: Path) -> float:
    """min pop_min_TgtBal over the run's whole convergence trace."""
    t = pd.read_parquet(run / "convergence.parquet",
                        columns=["generation", "pop_min_TgtBal"])
    n = len(t)
    assert t.generation.iloc[0] == 0 and t.generation.iloc[-1] == n - 1, \
        f"{run.name}: generation index is not 0..{n - 1}"
    return float(t.pop_min_TgtBal.min())


def _stats(run: Path) -> dict:
    import json
    return json.loads((run / "stats.json").read_text())


def load_exp24() -> dict[str, np.ndarray]:
    """Running floor per modality, generations 0..299."""
    empty = RUNS / "Exp-24" / EXP24_EMPTY
    assert empty.is_dir() and not list(empty.iterdir()), \
        f"{EXP24_EMPTY} is no longer the empty shell the note claims"
    qwen = RUNS / "Exp-24" / EXP24_QWEN
    assert qwen.is_dir() and not (qwen / "stats.json").exists(), \
        f"{EXP24_QWEN} now has a stats.json; the exclusion note is stale"

    out = {}
    for mode in MODES:
        cands = sorted(p.parent for p in (RUNS / "Exp-24").glob(
            f"exp24_llava_ov_{mode}_seed_83_*/convergence.parquet"))
        assert len(cands) == 1, f"{mode}: {len(cands)} usable run dirs {cands}"
        run = cands[0]

        cfg = yaml.safe_load(
            (CONFIGS / "Exp-24" / f"llava_ov_{mode}.yaml").read_text())
        assert cfg["modality"] == mode, f"{mode}: config modality {cfg['modality']}"
        assert (cfg["generations"], cfg["pop_size"]) == (300, 30), \
            f"{mode}: config budget {cfg['generations']}x{cfg['pop_size']}"

        s = _stats(run)
        assert (s["generations"], s["pop_size"]) == (300, 30), run.name
        assert s["pair"] == PAIR and s["seed_idx"] == 83, run.name
        assert s["model_id"] == LLAVA, run.name

        t = pd.read_parquet(run / "convergence.parquet",
                            columns=["generation", "pop_min_TgtBal"])
        assert len(t) == 300 and t.generation.iloc[0] == 0 \
            and t.generation.iloc[-1] == 299, f"{run.name}: unexpected gen index"
        c = t.pop_min_TgtBal.cummin().to_numpy()

        got = np.array([c[g] for g in SUPPORT_GENS])
        want = np.array(SUPPORT[mode])
        assert np.allclose(got, want, atol=5e-5, rtol=0), \
            f"{mode}: support points {got} != published {want}"
        out[mode] = c
        print(f"[a] {mode:11s} {run.name}  floor {c[-1]:.6g}  "
              f"support " + " ".join(f"{v:.4f}" for v in got))
    return out


def load_exp26() -> dict[tuple[str, int], float]:
    d26 = RUNS / "Exp-26"
    for name in EXP26_ABORTED:
        p = d26 / name
        assert p.is_dir() and not list(p.iterdir()), \
            f"{name} is no longer an empty aborted directory"
    stale, keep = d26 / EXP26_STALE, d26 / EXP26_RUN[("vqgan_baseline", 1)]
    assert stale.is_dir() and (stale / "stats.json").exists(), EXP26_STALE
    assert int(stale.name.rsplit("_", 1)[1]) < int(keep.name.rsplit("_", 1)[1]), \
        "the kept vqgan_baseline seed-1 run is not the later one"
    assert abs(_floor(stale) - 2.6226e-06) < 1e-9, \
        "the stale duplicate no longer reads 2.6e-6; the note is stale"
    qwen = d26 / EXP26_QWEN
    assert qwen.is_dir() and _stats(qwen)["model_id"] == QWEN, EXP26_QWEN

    cfgs = {b: yaml.safe_load((CONFIGS / "Exp-26" / "ov" /
                               f"{BACKEND_CFG[b]}.yaml").read_text())
            for b in BACKENDS}
    for b, cfg in cfgs.items():
        assert cfg["modality"] == "image_only", f"{b}: modality {cfg['modality']}"
        assert (cfg["generations"], cfg["pop_size"]) == (100, 20), b
    assert cfgs["vqgan_baseline"]["image"]["cone_filter"]["enabled"] is False
    assert cfgs["vqgan_cone"]["image"]["cone_filter"]["alpha_deg"] == 20.0
    assert cfgs["stylegan"]["image"]["backend"] == "stylegan_xl"
    assert cfgs["vqgan_baseline"]["image"]["backend"] == "vqgan_codebook"

    out, pairs = {}, {}
    for b in BACKENDS:
        for sd in SEEDS:
            run = d26 / EXP26_RUN[(b, sd)]
            s = _stats(run)
            assert (s["generations"], s["pop_size"]) == (100, 20), run.name
            assert s["model_id"] == LLAVA and s["seed_idx"] == sd, run.name
            assert s["seed_selection_mode"] == "gap_filter", run.name
            pairs.setdefault(sd, s["pair"])
            assert s["pair"] == pairs[sd], f"seed {sd}: pair differs across arms"
            f = _floor(run)
            want = EXP26_FLOOR[(b, sd)]
            assert abs(f - want) <= max(5e-7, 5e-4 * want), \
                f"{run.name}: floor {f!r} != published {want}"
            out[(b, sd)] = f
    assert pairs[83] == PAIR, "seed 83 is not the panel-(a) pair"
    for sd in SEEDS:
        print(f"[b] seed {sd:2d}  {' -> '.join(pairs[sd])}  " +
              "  ".join(f"{BACKEND_LABEL[b]} {out[(b, sd)]:.6g}"
                        for b in BACKENDS))
    return out


def _pareto_floor(run: Path) -> tuple[float, int]:
    """min TgtBal over the archived pareto_*.json fitness vectors."""
    import json
    files = sorted(run.glob("pareto_*.json"),
                   key=lambda p: int(p.stem.split("_")[1]))
    fit = np.array([json.loads(p.read_text())["fitness"] for p in files])
    assert fit.ndim == 2 and fit.shape[1] == 2, f"{run.name}: fitness shape"
    return float(fit[:, 1].min()), len(files)


def load_exp27() -> dict[str, float]:
    d27 = RUNS / "Exp-27"
    for name in EXP27_ABORTED:
        p = d27 / name
        assert p.is_dir() and not (p / "stats.json").exists(), \
            f"{name} now has a stats.json; the exclusion note is stale"
        assert len(pd.read_parquet(p / "convergence.parquet")) < 10, \
            f"{name} is no longer a short abort"

    out = {}
    for arm in ARMS:
        run = d27 / EXP27_RUN[arm]
        cfg = yaml.safe_load(
            (CONFIGS / "Exp-27" / f"qwen_mps_pairA_{arm}.yaml").read_text())
        cone = cfg["image"]["cone_filter"]
        if ARM_ALPHA[arm] is None:
            assert cone["enabled"] is False, f"{arm}: cone is enabled"
        else:
            assert cone["enabled"] is True and \
                cone["alpha_deg"] == ARM_ALPHA[arm], f"{arm}: alpha_deg"
        assert cfg["seeds"]["mode"] == "roster", arm
        assert cfg["seeds"]["roster"]["class_list"] == PAIR, arm
        assert (cfg["generations"], cfg["pop_size"]) == (100, 20), arm
        assert cfg["sut"]["model_id"] == QWEN, arm

        s = _stats(run)
        assert s["pair"] == PAIR and s["seed_selection_mode"] == "roster", run.name
        assert (s["generations"], s["pop_size"]) == (100, 20), run.name
        assert s["model_id"] == QWEN, run.name

        pf, npf = _pareto_floor(run)
        if arm == EXP27_BOUNDED:
            assert npf == EXP27_N_PARETO, \
                f"{arm}: {npf} pareto files, expected {EXP27_N_PARETO}"
            try:
                pd.read_parquet(run / "convergence.parquet")
            except Exception:
                pass
            else:
                raise AssertionError(
                    f"{arm}: convergence.parquet is readable again; the "
                    "pareto-json fallback and the hollow marker are stale")
            f = pf
        else:
            # The fallback is only trustworthy because it is exact wherever it
            # can be checked; assert that on every intact arm.
            f = _floor(run)
            assert pf == f, f"{arm}: pareto floor {pf!r} != trace floor {f!r}"
        want = EXP27_FLOOR[arm]
        assert abs(f - want) < 5e-5, f"{arm}: floor {f!r} != published {want}"
        out[arm] = f
        src = f"{npf} pareto fronts (upper bound)" if arm == EXP27_BOUNDED \
            else "convergence trace"
        print(f"[c] {arm:8s} alpha={str(ARM_ALPHA[arm]):5s}  floor {f:.6f}  "
              f"{run.name}  <- {src}")
    return out


# ---------------------------------------------------------------------------
# panels
# ---------------------------------------------------------------------------
def head(ax, label: str, note: str) -> None:
    ax.text(0.0, 1.012, label, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=FS_LABEL, fontweight="bold", color="#1A1A1A")
    ax.text(1.0, 1.012, note, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=FS_ANN, color="0.42")


def panel_a(ax, curves) -> None:
    g = np.arange(300)
    for mode in MODES:
        ax.plot(g, curves[mode], color=MODE_COLOR[mode], lw=1.35,
                ls=MODE_DASH[mode], solid_joinstyle="round", zorder=3)
    for mode, dy in (("image_only", 5), ("text_only", 5), ("joint", 5)):
        ax.annotate(MODE_LABEL[mode], xy=(299, curves[mode][-1]),
                    xytext=(-1, dy), textcoords="offset points", ha="right",
                    va="bottom", fontsize=FS_ANN, color=MODE_COLOR[mode],
                    zorder=4)

    ax.set_yscale("log")
    ax.set_xlim(-6, 306)
    ax.set_ylim(1e-4, 6.0)
    ax.set_xticks([0, 100, 200, 300])
    ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0])
    ax.set_yticklabels([r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$",
                        r"$10^{-1}$", r"$10^{0}$"])
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel("generation", fontsize=FS_LABEL, labelpad=2)
    ax.set_ylabel("running floor (TgtBal)", fontsize=FS_LABEL, labelpad=3)


def panel_b(ax, floors) -> None:
    xs = np.arange(len(SEEDS), dtype=float)
    off, bw = 0.27, 0.255
    for k, b in enumerate(BACKENDS):
        h = np.array([floors[(b, sd)] for sd in SEEDS])
        crossed = h <= CROSS_AT
        ax.bar(xs + (k - 1) * off, h, width=bw, color=BACKEND_COLOR[b],
               linewidth=0, zorder=3)
        if crossed.any():
            ax.bar(xs[crossed] + (k - 1) * off, h[crossed], width=bw,
                   facecolor="none", edgecolor=BACKEND_HATCH_EC[b],
                   linewidth=0.0, hatch="///", zorder=3.4)

    ax.axhline(CROSS_AT, color="0.45", lw=0.8, ls=(0, (4, 2.5)), zorder=2)
    ax.text(-0.52, CROSS_AT * 1.35, r"$10^{-2}$", ha="left", va="bottom",
            fontsize=FS_ANN, color="0.45", zorder=4)

    ax.set_yscale("log")
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(5e-6, 30.0)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(s) for s in SEEDS])
    ax.set_yticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1])
    ax.set_yticklabels([r"$10^{-5}$", "", r"$10^{-3}$", "", r"$10^{-1}$", "",
                        r"$10^{1}$"])
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel("seed", fontsize=FS_LABEL, labelpad=2)
    ax.set_ylabel("floor (min TgtBal)", fontsize=FS_LABEL, labelpad=3)

    handles = [Patch(facecolor=BACKEND_COLOR[b], linewidth=0,
                     label=BACKEND_LABEL[b]) for b in BACKENDS]
    handles.append(Patch(facecolor="white", edgecolor="0.35", linewidth=0.5,
                         hatch="///", label="crossed"))
    leg = ax.legend(handles=handles, loc="upper left", ncol=2, frameon=True,
                    framealpha=0.95, edgecolor="0.80", fancybox=False,
                    fontsize=FS_ANN, handlelength=0.95, handleheight=0.9,
                    handletextpad=0.35, labelspacing=0.30, columnspacing=0.55,
                    borderpad=0.32, borderaxespad=0.2)
    leg.get_frame().set_linewidth(0.5)
    return leg


def panel_c(ax, floors) -> None:
    xs = np.arange(len(ARMS), dtype=float)
    ys = np.array([floors[a] for a in ARMS])
    base = floors["baseline"]
    ib = ARMS.index(EXP27_BOUNDED)

    ax.axhline(base, color=C_REF, lw=0.8, ls=(0, (4, 2.5)), zorder=1.5)
    ax.text(len(ARMS) - 0.6, base + 0.006, "cone off", ha="right", va="bottom",
            fontsize=FS_ANN, color=C_REF, zorder=4)

    ax.plot(xs[1:], ys[1:], color=C_CONE, lw=1.1, alpha=0.55, zorder=2.5)
    solid = [i for i in range(1, len(ARMS)) if i != ib]
    ax.scatter(xs[solid], ys[solid], s=26, color=C_CONE, linewidths=0,
               zorder=3.5)
    ax.scatter([xs[ib]], [ys[ib]], s=30, facecolors="white", edgecolors=C_CONE,
               linewidths=1.0, zorder=3.6)
    ax.annotate("upper\nbound", xy=(xs[ib], ys[ib]), xytext=(0, 6),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=FS_ANN, color=C_CONE, linespacing=1.05, zorder=4)
    ax.scatter([xs[0]], [ys[0]], s=26, color=C_REF, linewidths=0, zorder=3.5)

    ax.set_xlim(-0.45, len(ARMS) - 0.55)
    ax.set_ylim(2.36, 2.75)
    ax.set_xticks(xs)
    ax.set_xticklabels(["off"] + [f"{int(ARM_ALPHA[a])}°"
                                  for a in ARMS[1:]])
    ax.set_yticks([2.4, 2.5, 2.6, 2.7])
    ax.set_xlabel(r"cone half-angle $\alpha$", fontsize=FS_LABEL, labelpad=2)
    ax.set_ylabel("floor (min TgtBal)", fontsize=FS_LABEL, labelpad=3)


def main() -> None:
    setup()
    curves = load_exp24()
    f26 = load_exp26()
    f27 = load_exp27()

    n_cross = sum(v <= CROSS_AT for v in f26.values())
    print(f"[b] crossings (floor <= {CROSS_AT:g}): {n_cross}/9 bars")
    print(f"[c] cone-off {f27['baseline']:.4f}; every alpha is worse by "
          + ", ".join(f"{f27[a] - f27['baseline']:+.4f}" for a in ARMS[1:]))

    fig = plt.figure(figsize=(W, H))
    axa = fig.add_axes(rect(X_A, BOTTOM, AXW_A, AXH, W=W, H=H))
    axb = fig.add_axes(rect(X_B, BOTTOM, AXW_B, AXH, W=W, H=H))
    axc = fig.add_axes(rect(X_C, BOTTOM, AXW_C, AXH, W=W, H=H))

    panel_a(axa, curves)
    leg = panel_b(axb, f26)
    panel_c(axc, f27)

    head(axa, "(a) modality", "LLaVA-NeXT")
    head(axb, "(b) backend", "LLaVA-NeXT")
    head(axc, "(c) cone", "Qwen3.5-4B")

    for ax in (axa, axb, axc):
        ax.tick_params(labelsize=FS_TICK, length=2.2, pad=2.0)
        ax.grid(True, axis="y", which="major", color=C_GRID, lw=0.5, zorder=0.2)
        ax.grid(False, axis="x")
        ax.set_axisbelow(True)
    axa.grid(True, axis="x", which="major", color=C_GRID, lw=0.5, zorder=0.2)

    # The panel-(b) legend must stay inside its own axes and clear the tallest
    # bar; both are geometry, so both are checked rather than eyeballed.
    fig.canvas.draw()
    lb = leg.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    ab = axb.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    tall = max(f26.values())
    y_bar = axb.transData.transform((0, tall))[1]
    y_leg = leg.get_window_extent().y0
    print(f"[layout] (b) legend {lb.width:.3f}x{lb.height:.3f} in, panel "
          f"{ab.width:.3f} in, headroom above tallest bar "
          f"{(y_leg - y_bar) / fig.dpi:.3f} in")
    assert lb.x0 >= ab.x0 - 1e-6 and lb.x1 <= ab.x1 + 1e-6, \
        "the panel-(b) legend overflows its axes"
    assert y_leg > y_bar, "the panel-(b) legend overlaps the tallest bar"

    save(fig, SLUG)


if __name__ == "__main__":
    main()
