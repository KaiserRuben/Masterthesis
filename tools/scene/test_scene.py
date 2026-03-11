#!/usr/bin/env python3
"""
Tests for the scene classification package.

Run with:
    python -m scene.test_scene           # Unit tests only
    python -m scene.test_scene --live    # Include live Ollama tests
"""

import json
import sys
from pathlib import Path


def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")

    # Enums
    from scene.enums import (
        RoadType, Weather, TimeOfDay, TrafficSituation,
        OcclusionLevel, DepthComplexity, VisualDegradation,
        SafetyCriticality, VulnerableRoadUserType,
    )

    # Response Models
    from scene.responses import (
        RESPONSE_MODELS, get_response_model, get_schema,
        WeatherResponse, RoadTypeResponse, TrafficSituationResponse,
        SpatialRelationsResponse, VulnerableRoadUsersResponse,
    )

    # Keys
    from scene.keys import (
        STAGE1_PROMPT, KEY_PROMPTS,
        KEYS, KEYS_ORIGINAL, KEYS_EXTENDED,
        get_prompt, get_keys_by_category, get_keys_by_difficulty,
    )

    # Top-level
    from scene import (
        Weather, get_prompt, get_schema, get_response_model, KEYS, get_tier_for_key,
    )

    print(f"  - All imports successful")
    print(f"  - Total keys: {len(KEYS)}")
    print(f"  - Original keys: {len(KEYS_ORIGINAL)}")
    print(f"  - Extended keys: {len(KEYS_EXTENDED)}")
    print(f"  - Response models: {len(RESPONSE_MODELS)}")
    return True


def test_enums():
    """Test enum functionality."""
    print("Testing enums...")

    from scene.enums import Weather, SafetyCriticality, VulnerableRoadUserType

    # Test string enum behavior
    assert Weather.clear.value == "clear"
    assert Weather("clear") == Weather.clear

    # Test criticality ordering
    criticality_order = [
        SafetyCriticality.tier4_minor,
        SafetyCriticality.tier3_moderate,
        SafetyCriticality.tier2_severe,
        SafetyCriticality.tier1_catastrophic,
    ]
    assert len(criticality_order) == 4

    # Test VRU types
    vru_types = list(VulnerableRoadUserType)
    assert VulnerableRoadUserType.pedestrian_child in vru_types
    assert VulnerableRoadUserType.animal in vru_types

    print(f"  - Enum tests passed")
    return True


def test_response_models():
    """Test Pydantic response model functionality."""
    print("Testing response models...")

    from scene.responses import (
        WeatherResponse, RoadTypeResponse, TrafficSituationResponse,
        TrafficSituationPoints, SpatialRelationsResponse, SpatialRelation,
        VehicleCountByTypeResponse, PedestrianCountResponse,
    )
    from scene.enums import Weather, RoadType, TrafficSituation, Confidence

    # Test WeatherResponse
    weather = WeatherResponse(reasoning="Clear sky", weather=Weather.clear)
    assert weather.weather == Weather.clear
    assert weather.reasoning == "Clear sky"

    # Test RoadTypeResponse
    road = RoadTypeResponse(reasoning="Multiple lanes", road_type=RoadType.highway)
    assert road.road_type == RoadType.highway

    # Test TrafficSituationResponse
    points = TrafficSituationPoints(
        vehicles=2, pedestrians=2, construction=0,
        intersection=2, signals=1, weather=0, visibility=0
    )
    traffic = TrafficSituationResponse(
        points=points,
        total=7,
        category=TrafficSituation.complex
    )
    assert traffic.category == TrafficSituation.complex
    assert traffic.total == 7

    # Test SpatialRelationsResponse
    relations = SpatialRelationsResponse(
        relations=[
            SpatialRelation(
                object_a="car_1",
                object_b="pedestrian_1",
                relation="in_front_of",
                confidence=Confidence.high
            )
        ]
    )
    assert len(relations.relations) == 1
    assert relations.relations[0].confidence == Confidence.high

    # Test VehicleCountByTypeResponse
    vehicles = VehicleCountByTypeResponse(
        reasoning="Counted vehicles",
        cars=5, suvs_trucks=3, commercial=1, motorcycles=0, other=1,
        confidence=Confidence.high
    )
    assert vehicles.total == 10

    # Test PedestrianCountResponse
    peds = PedestrianCountResponse(
        reasoning="Some people visible",
        count=5, confidence=Confidence.medium, occluded_estimate=2
    )
    assert peds.count == 5

    print(f"  - Response model tests passed")
    return True


def test_model_validate_json():
    """Test parsing JSON directly to Pydantic models."""
    print("Testing model_validate_json...")

    from scene.responses import WeatherResponse, TrafficSituationResponse
    from scene.enums import Weather, TrafficSituation

    # Test WeatherResponse from JSON
    weather_json = '{"reasoning": "Clear blue sky", "weather": "clear"}'
    weather = WeatherResponse.model_validate_json(weather_json)
    assert weather.weather == Weather.clear

    # Test TrafficSituationResponse from JSON
    traffic_json = '''
    {
        "points": {
            "vehicles": 1, "pedestrians": 0, "construction": 0,
            "intersection": 0, "signals": 0, "weather": 0, "visibility": 0
        },
        "total": 1,
        "category": "simple"
    }
    '''
    traffic = TrafficSituationResponse.model_validate_json(traffic_json)
    assert traffic.category == TrafficSituation.simple
    assert traffic.total == 1

    print(f"  - model_validate_json tests passed")
    return True


def test_model_dump():
    """Test serializing Pydantic models back to JSON."""
    print("Testing model_dump...")

    from scene.responses import WeatherResponse, SpatialRelationsResponse, SpatialRelation
    from scene.enums import Weather, Confidence

    # Test WeatherResponse serialization
    weather = WeatherResponse(reasoning="Clear sky", weather=Weather.clear)
    dumped = weather.model_dump(mode="json")
    assert dumped["weather"] == "clear"
    assert dumped["reasoning"] == "Clear sky"

    # Test nested model serialization
    relations = SpatialRelationsResponse(
        relations=[
            SpatialRelation(
                object_a="car",
                object_b="pedestrian",
                relation="left_of",
                confidence=Confidence.high
            )
        ]
    )
    dumped = relations.model_dump(mode="json")
    assert dumped["relations"][0]["confidence"] == "high"

    # Verify it's JSON serializable
    json_str = json.dumps(dumped)
    assert "left_of" in json_str

    print(f"  - model_dump tests passed")
    return True


def test_keys():
    """Test key registry functionality."""
    print("Testing keys...")

    from scene.keys import (
        KEYS, KEYS_ORIGINAL, KEYS_EXTENDED,
        KEY_PROMPTS, get_prompt, get_keys_by_category,
        get_keys_by_difficulty, get_tier_for_key,
    )
    from scene.responses import RESPONSE_MODELS, get_schema

    # Test key counts
    assert len(KEYS) == len(KEY_PROMPTS)
    assert len(KEYS) == len(RESPONSE_MODELS)
    assert len(KEYS_ORIGINAL) == 10
    assert len(KEYS_EXTENDED) == len(KEYS) - 10

    # Test get_prompt
    weather_prompt = get_prompt("weather")
    assert "WEATHER" in weather_prompt
    assert "clear" in weather_prompt

    # Test get_schema (now from Pydantic models)
    weather_schema = get_schema("weather")
    assert weather_schema["type"] == "object"
    assert "properties" in weather_schema

    # Test categories
    scene_context = get_keys_by_category("scene_context")
    assert "weather" in scene_context
    assert "road_type" in scene_context

    safety_critical = get_keys_by_category("safety_critical")
    assert "safety_criticality" in safety_critical
    assert "vulnerable_road_users" in safety_critical

    # Test difficulty
    easy_keys = get_keys_by_difficulty("easy")
    assert "weather" in easy_keys

    critical_keys = get_keys_by_difficulty("critical")
    assert "safety_criticality" in critical_keys

    # Test tier mapping
    assert get_tier_for_key("weather") == "small"
    assert get_tier_for_key("safety_criticality") == "large"

    print(f"  - Key tests passed")
    print(f"  - Categories: {len(get_keys_by_category('scene_context'))} scene_context, "
          f"{len(get_keys_by_category('spatial_reasoning'))} spatial_reasoning, "
          f"{len(get_keys_by_category('safety_critical'))} safety_critical")
    return True


def test_schema_from_models():
    """Test that schemas generated from Pydantic models are valid."""
    print("Testing schema generation from models...")

    from scene.responses import RESPONSE_MODELS, get_schema
    from scene.keys import KEYS

    for key in KEYS:
        model = RESPONSE_MODELS[key]
        schema = get_schema(key)

        # Must be object type
        assert schema.get("type") == "object", f"{key}: must be object type"

        # Must have properties
        assert "properties" in schema, f"{key}: must have properties"

        # Model class should match
        assert model.model_json_schema() == schema, f"{key}: schema mismatch"

    print(f"  - All {len(KEYS)} schemas generated correctly from models")
    return True


def test_vlm_integration():
    """Test integration with vlm package."""
    print("Testing VLM integration...")

    # Add tools to path
    tools_path = Path(__file__).parent.parent
    if str(tools_path) not in sys.path:
        sys.path.insert(0, str(tools_path))

    from vlm.config import DEFAULT_KEY_TIERS, get_model_tiers, resolve_tier
    from scene.keys import KEYS

    # Check all keys have tier mappings
    missing = [k for k in KEYS if k not in DEFAULT_KEY_TIERS]
    if missing:
        print(f"  WARNING: Keys without tier mapping: {missing}")
    else:
        print(f"  - All {len(KEYS)} keys have tier mappings")

    # Test tier resolution
    tiers = get_model_tiers()
    assert "small" in tiers
    assert "medium" in tiers
    assert "large" in tiers

    # Test model resolution for keys
    for key in ["weather", "safety_criticality", "spatial_relations"]:
        tier = DEFAULT_KEY_TIERS.get(key, "medium")
        model = resolve_tier(tier)
        print(f"    {key}: {tier} -> {model}")

    print(f"  - VLM integration tests passed")
    return True


def test_live_inference(model: str = "qwen3-vl:8b"):
    """Test live inference with Ollama (requires running server)."""
    print(f"Testing live inference with {model}...")

    try:
        from ollama import Client
        client = Client()

        from scene.keys import get_prompt
        from scene.responses import get_schema, get_response_model

        prompt = get_prompt("weather")
        schema = get_schema("weather")
        response_model = get_response_model("weather")

        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "SCENE: Clear blue sky, bright sunshine, dry road surfaces, sharp shadows."}
            ],
            format=schema,
            options={"num_ctx": 8192}
        )

        # Parse directly to Pydantic model
        result = response_model.model_validate_json(response.message.content)

        print(f"  - Model response: {result.model_dump()}")
        print(f"  - Weather: {result.weather}")
        print(f"  - Live inference test passed")
        return True

    except Exception as e:
        print(f"  - Live inference failed: {e}")
        return False


def main():
    """Run all tests."""
    import argparse

    parser = argparse.ArgumentParser(description="Scene classification tests")
    parser.add_argument("--live", action="store_true", help="Include live Ollama tests")
    parser.add_argument("--model", default="qwen3-vl:8b", help="Model for live tests")
    args = parser.parse_args()

    # Add tools to path
    tools_path = Path(__file__).parent.parent
    if str(tools_path) not in sys.path:
        sys.path.insert(0, str(tools_path))

    print("=" * 60)
    print("Scene Classification Package Tests")
    print("=" * 60)
    print()

    tests = [
        ("Imports", test_imports),
        ("Enums", test_enums),
        ("Response Models", test_response_models),
        ("model_validate_json", test_model_validate_json),
        ("model_dump", test_model_dump),
        ("Keys", test_keys),
        ("Schema from Models", test_schema_from_models),
        ("VLM Integration", test_vlm_integration),
    ]

    if args.live:
        tests.append(("Live Inference", lambda: test_live_inference(args.model)))

    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
            print()
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
            print()

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {name}")

    print()
    print(f"Passed: {passed}/{total}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
