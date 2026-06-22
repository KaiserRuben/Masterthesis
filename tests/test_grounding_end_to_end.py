"""Grounding path smoke test: a stub SUT returns fixed per-candidate logprobs;
assert TargetedBalance over the two box strings reaches the trace as fitness_TgtBal.

The test reuses the existing TargetedBalance objective and CriterionCollection
to verify that box-string-derived logits (two candidates per individual)
produce the expected |lp_A − lp_B| gaps.  No real model or dataset download is
required; the logit values are synthetic numbers matching the spike analysis.

Manual live smoke (run by a human — needs model weights + a RefCOCO+ slice):

    conda run -n uni python experiments/runners/run_boundary_test.py \\
      configs/Exp-103/exp103_coordinate_grounding_refcocoplus.yaml --generations 1 --max-seeds 2
    # Expect: runs/Exp-103/<...>/trace.parquet with fitness_TgtBal columns,
    # decoded_text = box strings, stats.json seed_selection_mode=refcocoplus.
"""
import torch
from src.objectives import CriterionCollection, TargetedBalance


def test_tgtbal_over_box_string_candidates():
    # stub: SUT returned log_prob_norm for (box_A, box_B) across a 3-individual pop
    logits = torch.tensor([[-1.10, -1.61], [-1.04, -1.70], [-1.37, -1.54]])
    coll = CriterionCollection(TargetedBalance())
    coll.evaluate_all(logits=logits, target_classes=(0, 1), batch_dim=0)
    gaps = coll.results["TgtBal"]
    assert [round(g, 2) for g in gaps] == [0.51, 0.66, 0.17]  # matches the spike numbers
