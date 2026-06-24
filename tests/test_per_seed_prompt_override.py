from PIL import Image
from src.config import SeedTriple, ExperimentConfig
from src.evolutionary.vlm_boundary_tester import effective_prompt_template


def test_uses_seed_prompt_when_present():
    seed = SeedTriple(image=Image.new("RGB", (8, 8)), class_a="[1,2,3,4]",
                      class_b="[5,6,7,8]", metadata={"prompt_template": "Locate the dog."})
    assert effective_prompt_template(seed, ExperimentConfig()) == "Locate the dog."


def test_falls_back_to_config_prompt():
    seed = SeedTriple(image=Image.new("RGB", (8, 8)), class_a="a", class_b="b")
    cfg = ExperimentConfig(prompt_template="What is this?")
    assert effective_prompt_template(seed, cfg) == "What is this?"
