from PIL import Image
from src.config import ExperimentConfig, SeedConfig, RefCocoPlusConfig, SeedTriple
import src.common.pipeline_bootstrap as pb


def test_dispatch_routes_to_refcocoplus(monkeypatch):
    fake = [SeedTriple(image=Image.new("RGB", (8, 8)), class_a="[1, 2, 3, 4]",
                       class_b="[5, 6, 7, 8]", metadata={"prompt_template": "Locate the cat."})]
    monkeypatch.setattr(pb, "refcocoplus_seeds", lambda sut, cfg, ds: fake, raising=False)
    exp = ExperimentConfig(modality="grounding",
                           seeds=SeedConfig(mode="refcocoplus",
                                            refcocoplus=RefCocoPlusConfig(data_root="x")))
    class _C: sut = None; data_source = None
    out = pb.prepare_pipeline_seeds(_C(), exp)
    assert out is fake
