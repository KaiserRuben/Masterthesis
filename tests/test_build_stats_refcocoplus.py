# tests/test_build_stats_refcocoplus.py
from PIL import Image
from src.config import (ExperimentConfig, SeedConfig, RefCocoPlusConfig,
                        GroundingConfig, SeedTriple)
from src.evolutionary.vlm_boundary_tester import build_stats

class _FakeManip:
    gene_bounds = __import__("numpy").zeros(1); image_dim = 1; text_dim = 0
    def __getattr__(self, n): return 0

def test_stats_record_refcocoplus_provenance():
    cfg = ExperimentConfig(
        modality="grounding",
        grounding=GroundingConfig(coordinate_space="norm_1000"),
        seeds=SeedConfig(mode="refcocoplus",
                         refcocoplus=RefCocoPlusConfig(data_root="x", split="testA")))
    seed = SeedTriple(image=Image.new("RGB", (8, 8)), class_a="[1, 2, 3, 4]",
                      class_b="[5, 6, 7, 8]",
                      metadata={"prompt_template": "Locate the cat.", "ref_id_a": 1})
    stats = build_stats(0, seed, cfg, _FakeManip(), 0, 0.0,
                        ("[1, 2, 3, 4]", "[5, 6, 7, 8]"),
                        ("[1, 2, 3, 4]", "[5, 6, 7, 8]"), (0, 1),
                        {"hits": 0, "misses": 0})
    assert stats["seed_selection_mode"] == "refcocoplus"
    assert stats["refcoco_split"] == "testA"
    assert stats["coordinate_space"] == "norm_1000"
    assert stats["seed_metadata"]["ref_id_a"] == 1
