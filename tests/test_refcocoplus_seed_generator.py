from PIL import Image
from src.common.refcocoplus_seed_generator import normalize_box, build_seed_triples


def test_normalize_box_norm_1000():
    # box in pixels on a 640x480 image -> [0,1000] ints
    assert normalize_box((64, 48, 320, 240), 640, 480, "norm_1000") == "[100, 100, 500, 500]"


def test_normalize_box_abs_pixels_passthrough():
    assert normalize_box((10, 20, 30, 40), 640, 480, "abs_pixels") == "[10, 20, 30, 40]"


def test_build_seed_triples_pairs_same_category():
    img = Image.new("RGB", (640, 480), "white")
    # two 'person' referents on one image
    items = [{
        "image": img, "image_id": 7, "image_w": 640, "image_h": 480,
        "referent": "person",
        "ref_a": {"ref_id": 1, "bbox_px": (64, 48, 320, 240)},
        "ref_b": {"ref_id": 2, "bbox_px": (384, 48, 576, 240)},
    }]
    seeds = build_seed_triples(items, coordinate_space="norm_1000")
    assert len(seeds) == 1
    s = seeds[0]
    assert s.class_a == "[100, 100, 500, 500]"
    assert s.class_b == "[600, 100, 900, 500]"
    assert s.metadata["prompt_template"] == "Locate the person."
    assert s.metadata["coordinate_space"] == "norm_1000"
    assert s.metadata["ref_id_a"] == 1 and s.metadata["ref_id_b"] == 2
