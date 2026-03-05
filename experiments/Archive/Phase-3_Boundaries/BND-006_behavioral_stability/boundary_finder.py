"""BND-006: VLM Behavioral Stability — Pareto-front boundary search.

3-objective NSGA-II optimization to find manipulated images that:
  1. Stay perceptually close to the original  (minimize LPIPS)
  2. Cause large trajectory deviations         (maximize ADE vs GT)
  3. Cause high output variance across k runs  (maximize Var(ADE))
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from tqdm import tqdm

_TOOLS = Path(__file__).resolve().parents[3] / "tools"
sys.path.insert(0, str(_TOOLS / "smoo"))
sys.path.insert(0, str(_TOOLS / "alpamayo" / "src"))
sys.path.insert(0, str(_TOOLS))


def _detect_device() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            _ = torch.tensor([1.0], device="mps") * 2
            return "mps", torch.float32
        except Exception:
            pass
    return "cpu", torch.float32

from pydantic import BaseModel
from src import SMOO
from src.manipulator import Manipulator
from src.manipulator._candidate import Candidate, CandidateList
from src.objectives import Criterion, CriterionCollection
from src.optimizer import PymooOptimizer
from src.sut import SUT


class SpeedClass(str, Enum):
    SLOW_DOWN = "SLOW_DOWN"
    STAY = "STAY"
    SPEED_UP = "SPEED_UP"


def classify_speed(recommended_ms: float, current_ms: float, threshold_kmh: float = 1.0) -> SpeedClass:
    diff = recommended_ms - current_ms
    t = threshold_kmh / 3.6
    if diff < -t: return SpeedClass.SLOW_DOWN
    if diff > t:  return SpeedClass.SPEED_UP
    return SpeedClass.STAY


@dataclass
class BoundaryResult:
    clip_id: str
    archive_intensities: list[NDArray]
    archive_lpips: list[float]
    archive_mean_ade: list[float]
    archive_ade_var: list[float]
    archive_metadata: list[dict]
    archive_generations: list[int]
    pareto_front: list[dict]
    generations: int
    query_count: int


@dataclass
class Manipulation:
    name: str
    fn: Callable[[Tensor, float], Tensor]
    intensity: float = 0.0


def _brightness(img: Tensor, t: float) -> Tensor: return img * (1.0 - t)
def _fog(img: Tensor, t: float) -> Tensor:        return img * (1.0 - t) + t
def _contrast(img: Tensor, t: float) -> Tensor:   return img * (1.0 - t) + img.mean(dim=-3, keepdim=True) * t

DEFAULT_MANIPULATIONS = [
    Manipulation("brightness", _brightness),
    Manipulation("fog", _fog),
    Manipulation("contrast", _contrast),
]


@dataclass
class ManipulationCandidate(Candidate):
    intensities: dict[str, float] = field(default_factory=dict)


class PixelManipulator(Manipulator):
    def __init__(self, manipulations: list[Manipulation]) -> None:
        self._manipulations = list(manipulations)

    def _apply_chain(self, img_f: Tensor, intensities: dict[str, float]) -> Tensor:
        for m in self._manipulations:
            v = intensities.get(m.name, m.intensity)
            if v > 0.0: img_f = m.fn(img_f, v)
        return img_f.clamp(0.0, 1.0)

    def manipulate(self, candidates: CandidateList[ManipulationCandidate], **kw: Any) -> Tensor:
        base = kw["images"].float() / 255.0
        batch = kw["intensities"]  # NDArray (pop_size, n_manipulations)
        out = []
        for row in batch:
            d = {m.name: float(row[j]) for j, m in enumerate(self._manipulations)}
            out.append(self._apply_chain(base, d))
        return (torch.stack(out) * 255).to(torch.uint8)

    def get_images(self, z: Tensor) -> Tensor:
        return z

    def apply(self, images: Tensor, intensities: dict[str, float]) -> Tensor:
        return (self._apply_chain(images.float() / 255.0, intensities) * 255).to(torch.uint8)

    def intensity_dict(self, vector: NDArray) -> dict[str, float]:
        return {m.name: float(vector[j]) for j, m in enumerate(self._manipulations)}


class AlpamayoSUT(SUT):
    def __init__(self, model: Any, processor: Any, device: str = "cuda",
                 dtype: torch.dtype = torch.bfloat16) -> None:
        self._model, self._processor = model, processor
        self._device, self._dtype, self._batch_size = torch.device(device), dtype, 1

    @staticmethod
    def _speed_from_xy(xy: np.ndarray, dt: float = 0.1) -> float:
        d = np.diff(xy, axis=0)
        return float(np.linalg.norm(d, axis=1).mean() / dt)

    def _prepare_model_inputs(self, inpt: dict) -> dict:
        from alpamayo_r1 import helper
        messages = helper.create_message(inpt["image_frames"].flatten(0, 1))
        inputs = self._processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False,
            continue_final_message=True, return_dict=True, return_tensors="pt")
        return helper.to_device({
            "tokenized_data": inputs,
            "ego_history_xyz": inpt["ego_history_xyz"],
            "ego_history_rot": inpt["ego_history_rot"],
        }, str(self._device))

    def _run_inference(self, model_inputs: dict) -> Tensor:
        ctx = torch.autocast("cuda", dtype=self._dtype) if self._device.type == "cuda" \
            else torch.inference_mode()
        with ctx:
            pred_xyz, _, _ = self._model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs, top_p=0.98, temperature=0.6,
                num_traj_samples=1, max_generation_length=256, return_extra=True)
        return pred_xyz

    def process_input_k(self, inpt: dict, k: int) -> dict:
        """Run k independent forward passes and return per-run trajectories."""
        hist = inpt["ego_history_xyz"][0, 0].cpu().numpy()
        current = self._speed_from_xy(hist[-5:, :2])
        model_inputs = self._prepare_model_inputs(inpt)
        # Snapshot tokenized_data — the model pops keys from it during inference
        _tok_snapshot = {**model_inputs["tokenized_data"]}

        pred_xyz_list: list[Tensor] = []
        speeds: list[float] = []
        for _ in range(k):
            run_inputs = {**model_inputs, "tokenized_data": {**_tok_snapshot}}
            pred_xyz = self._run_inference(run_inputs)
            pred_xyz_list.append(pred_xyz[0, 0, 0].cpu())  # (timesteps, 3)
            speed = self._speed_from_xy(pred_xyz[0, 0, 0, :10, :2].cpu().numpy())
            speeds.append(speed)

        mean_speed = float(np.mean(speeds))
        return {
            "pred_xyz_list": pred_xyz_list,
            "individual_speeds": speeds,
            "recommended_speed": mean_speed,
            "current_speed": current,
            "speed_class": classify_speed(mean_speed, current),
        }

    def process_input(self, inpt: Any) -> dict:
        """Single-pass wrapper (satisfies SUT ABC)."""
        return self.process_input_k(inpt, k=1)

    def input_valid(self, inpt: Any, cond: SpeedClass) -> tuple[bool, dict]:
        r = self.process_input(inpt)
        return r["speed_class"] == cond, r


# =============================================================================
# OLLAMA SUT (Qwen3-VL)
# =============================================================================

SPEED_PROMPT = """\
You are an autonomous driving safety system analyzing a sequence of \
front camera images from a moving vehicle. Based on the visual conditions (weather, \
visibility, road geometry, traffic, obstacles), classify what speed \
action the vehicle should take.

SLOW_DOWN: Hazards, reduced visibility, obstacles, curves, or any \
condition requiring deceleration.
MAINTAIN_SPEED: Clear road, normal conditions, no reason to change.
SPEED_UP: Clear road AND vehicle is below normal speed for conditions."""


class SpeedActionResponse(BaseModel):
    reasoning: str
    action: Literal["SLOW_DOWN", "MAINTAIN_SPEED", "SPEED_UP"]


_ACTION_TO_CLASS = {
    "SLOW_DOWN": SpeedClass.SLOW_DOWN,
    "MAINTAIN_SPEED": SpeedClass.STAY,
    "SPEED_UP": SpeedClass.SPEED_UP,
}


class OllamaSUT:
    """Qwen3-VL as a driving scene speed classifier via Ollama."""

    def __init__(self, model: str = "qwen3-vl:8b", n_samples: int = 1) -> None:
        from vlm import Message, create_provider, get_model

        self._provider = create_provider()
        self._model_config = get_model(model)
        self._Message = Message
        self._n_samples = n_samples

    def _build_messages(self, inpt: dict) -> list:
        import base64
        import io

        from PIL import Image

        frames = inpt["image_frames"]  # (N_cam, N_time, 3, H, W)
        front = frames[1]  # (N_time, 3, H, W)

        images: list[str] = []
        for t in range(front.shape[0]):
            img = Image.fromarray(front[t].permute(1, 2, 0).cpu().numpy())
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            images.append(base64.b64encode(buf.getvalue()).decode())

        return [
            self._Message("system", SPEED_PROMPT),
            self._Message(
                "user",
                "Analyze these driving scene images and classify the required speed action.",
                images=tuple(images),
            ),
        ]

    def process_input(self, inpt: dict) -> dict:
        from collections import Counter

        messages = self._build_messages(inpt)
        schema = SpeedActionResponse.model_json_schema()

        responses: list[SpeedActionResponse] = []
        prompt_tokens = 0
        completion_tokens = 0
        for _ in range(self._n_samples):
            result = self._provider.complete(
                messages=messages, model=self._model_config, json_schema=schema,
            )
            responses.append(SpeedActionResponse.model_validate_json(result.content))
            prompt_tokens += result.prompt_tokens or 0
            completion_tokens += result.completion_tokens or 0

        # Majority vote on action
        votes = Counter(r.action for r in responses)
        winner = votes.most_common(1)[0][0]

        # Keep all reasoning (one per sample)
        all_reasoning = [r.reasoning for r in responses]
        reasoning = all_reasoning[0] if len(all_reasoning) == 1 else \
            "\n---\n".join(f"[{i+1}/{len(all_reasoning)}] {r}" for i, r in enumerate(all_reasoning))

        return {
            "speed_class": _ACTION_TO_CLASS[winner],
            "recommended_speed": 0.0,
            "current_speed": 0.0,
            "reasoning": reasoning,
            "metadata": {
                "model": self._model_config.name,
                "n_samples": self._n_samples,
                "votes": dict(votes),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        }

    def input_valid(self, inpt: Any, cond: SpeedClass) -> tuple[bool, dict]:
        r = self.process_input(inpt)
        return r["speed_class"] == cond, r


# =============================================================================
# CRITERIA
# =============================================================================

def _compute_ade(pred: Tensor, gt: Tensor) -> float:
    """Average Displacement Error (xy) between two trajectory tensors."""
    n = min(len(pred), len(gt))
    return float(torch.norm(pred[:n, :2] - gt[:n, :2], dim=1).mean())


class LPIPSCriterion(Criterion):
    """Perceptual distance between original and manipulated frames (minimize)."""

    _name = "LPIPS"

    def __init__(self) -> None:
        super().__init__(allow_batched=True)
        import lpips
        self._lpips = lpips.LPIPS(net="alex")

    def evaluate(self, *, original_frames: Tensor, manipulated_frames_batch: list[Tensor],
                 **_: Any) -> list[float]:
        orig = original_frames.float() / 255.0
        scores: list[float] = []
        with torch.no_grad():
            for manip in manipulated_frames_batch:
                d = self._lpips(orig, manip.float() / 255.0, normalize=True)
                scores.append(float(d.mean()))
        return scores


class TrajectoryDeviationCriterion(Criterion):
    """Mean ADE across k runs (maximize via negation → minimize -ADE)."""

    _name = "TrajectoryDeviation"

    def __init__(self) -> None:
        super().__init__(allow_batched=True)

    def evaluate(self, *, gt_future_xyz: Tensor, pred_xyz_lists: list[list[Tensor]],
                 **_: Any) -> list[float]:
        gt = gt_future_xyz.squeeze()  # (timesteps, 3)
        scores: list[float] = []
        for pred_list in pred_xyz_lists:
            ades = [_compute_ade(p, gt) for p in pred_list]
            scores.append(-float(np.mean(ades)))
        return scores


class OutputVarianceCriterion(Criterion):
    """Variance of ADE across k runs (maximize via negation → minimize -Var)."""

    _name = "OutputVariance"

    def __init__(self) -> None:
        super().__init__(allow_batched=True)

    def evaluate(self, *, gt_future_xyz: Tensor, pred_xyz_lists: list[list[Tensor]],
                 **_: Any) -> list[float]:
        gt = gt_future_xyz.squeeze()  # (timesteps, 3)
        scores: list[float] = []
        for pred_list in pred_xyz_lists:
            ades = [_compute_ade(p, gt) for p in pred_list]
            scores.append(-float(np.var(ades)))
        return scores


# =============================================================================
# BOUNDARY FINDER
# =============================================================================

class BoundaryFinder(SMOO):
    def __init__(self, sut: AlpamayoSUT, manipulator: PixelManipulator,
                 optimizer: PymooOptimizer, objectives: CriterionCollection,
                 k_samples: int = 10) -> None:
        super().__init__(sut=sut, manipulator=manipulator, optimizer=optimizer,
                         objectives=objectives, restrict_classes=None, use_wandb=False)
        self._k_samples = k_samples

    def test(self, clip_id: str, scene_data: dict, *,
             max_generations: int = 50, stagnation_limit: int = 10) -> BoundaryResult:
        orig = scene_data["image_frames"]
        orig_flat = orig.flatten(0, 1)  # (N_cams*N_times, C, H, W)
        orig_shape = orig.shape
        gt_future_xyz = scene_data["ego_future_xyz"]  # (1, 1, timesteps, 3)
        opt: PymooOptimizer = self._optimizer

        archive_x: list[NDArray] = []
        archive_lpips: list[float] = []
        archive_mean_ade: list[float] = []
        archive_ade_var: list[float] = []
        archive_gen: list[int] = []
        archive_meta: list[dict] = []
        archive_seen: set[tuple[float, ...]] = set()

        prev_pareto_size = 0
        stagnation_count = 0
        query_count = 0
        final_gen = 0

        gen_bar = tqdm(range(max_generations), desc=f"  {clip_id}", unit="gen",
                       leave=True)
        for gen in gen_bar:
            final_gen = gen
            population = opt.get_x_current()  # (pop_size, n_manipulations)
            pop_size = population.shape[0]

            gen_manip_frames: list[Tensor] = []
            gen_pred_lists: list[list[Tensor]] = []
            gen_meta: list[dict] = []

            for i in tqdm(range(pop_size), desc="    queries", unit="q",
                          leave=False):
                ints_dict = self._manipulator.intensity_dict(population[i])
                modified = self._manipulator.apply(orig_flat, ints_dict)
                gen_manip_frames.append(modified)

                scene_mod = {**scene_data,
                             "image_frames": modified.reshape(orig_shape)}
                result = self._sut.process_input_k(scene_mod, self._k_samples)
                gen_pred_lists.append(result["pred_xyz_list"])
                gen_meta.append({
                    "speeds": result["individual_speeds"],
                    "mean_speed": result["recommended_speed"],
                })
                query_count += self._k_samples
                self._cleanup()

            # Evaluate all three objectives for the generation
            self._objectives.evaluate_all(
                original_frames=orig_flat,
                manipulated_frames_batch=gen_manip_frames,
                gt_future_xyz=gt_future_xyz,
                pred_xyz_lists=gen_pred_lists,
            )
            fitness = tuple(np.asarray(v) for v in self._objectives.results.values())
            opt.assign_fitness(fitness)
            opt.update()

            # Archive new individuals
            res = self._objectives.results
            lpips_v = res["LPIPS"]
            ade_v = res["TrajectoryDeviation"]
            var_v = res["OutputVariance"]

            for i in range(pop_size):
                key = tuple(population[i].tolist())
                if key not in archive_seen:
                    archive_seen.add(key)
                    archive_x.append(population[i])
                    archive_lpips.append(lpips_v[i])
                    archive_mean_ade.append(-ade_v[i])   # store positive
                    archive_ade_var.append(-var_v[i])     # store positive
                    archive_gen.append(gen)
                    archive_meta.append(gen_meta[i])

            # Stagnation: track Pareto front size
            pareto_size = len(opt.best_candidates)
            gen_bar.set_postfix(queries=query_count, pareto=pareto_size,
                                stag=stagnation_count)
            if pareto_size > prev_pareto_size:
                prev_pareto_size = pareto_size
                stagnation_count = 0
            else:
                stagnation_count += 1
            if stagnation_count >= stagnation_limit:
                gen_bar.set_postfix_str(
                    f"converged (stagnation={stagnation_limit})")
                break

        # Extract Pareto front from optimizer
        pareto_front: list[dict] = []
        for cand in opt.best_candidates:
            sol = cand.solution.reshape(len(DEFAULT_MANIPULATIONS))
            ints = self._manipulator.intensity_dict(sol)
            pareto_front.append({
                **ints,
                "lpips": float(cand.fitness[0]),
                "mean_ade": float(-cand.fitness[1]),
                "ade_variance": float(-cand.fitness[2]),
            })

        return BoundaryResult(
            clip_id=clip_id,
            archive_intensities=archive_x,
            archive_lpips=archive_lpips,
            archive_mean_ade=archive_mean_ade,
            archive_ade_var=archive_ade_var,
            archive_metadata=archive_meta,
            archive_generations=archive_gen,
            pareto_front=pareto_front,
            generations=final_gen + 1,
            query_count=query_count,
        )
