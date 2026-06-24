import pytest
from src.config import SeedConfig, RefCocoPlusConfig

def test_refcocoplus_mode_requires_block():
    with pytest.raises(ValueError, match="requires a seeds.refcocoplus"):
        SeedConfig(mode="refcocoplus")

def test_refcocoplus_mode_ok():
    sc = SeedConfig(mode="refcocoplus",
                    refcocoplus=RefCocoPlusConfig(data_root="~/.cache/refcoco"))
    assert sc.mode == "refcocoplus"
    assert sc.refcocoplus.split == "testA"

def test_refcocoplus_rejects_conflicting_blocks():
    from src.config import GapFilterConfig
    with pytest.raises(ValueError, match="drop one"):
        SeedConfig(mode="refcocoplus",
                   refcocoplus=RefCocoPlusConfig(data_root="x"),
                   gap_filter=GapFilterConfig())
