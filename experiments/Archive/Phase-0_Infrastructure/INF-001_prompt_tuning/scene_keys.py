"""
Scene Classification Keys - Backward Compatibility Wrapper

This module re-exports from tools/scene/ for backward compatibility.
The canonical definitions are now in tools/scene/.

For new code, import directly from scene:
    from scene import get_prompt, get_schema, KEYS
"""

import sys
from pathlib import Path

# Add tools to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

# Re-export everything from the scene package
from scene import (
    # Enums
    RoadType,
    Weather,
    TimeOfDay,
    TrafficSituation,
    OcclusionLevel,
    DepthComplexity,
    SpatialQueryType,
    VisualDegradation,
    SafetyCriticality,
    VulnerableRoadUserType,
    RequiredAction,
    HazardUrgency,
    TrafficLightState,
    LaneMarkingType,
    Confidence,
    # Models
    TrafficSituationPoints,
    TrafficSituationResult,
    SpatialRelation,
    DistanceEstimate,
    VRUInstance,
    HazardInstance,
    CountingResult,
    VehicleCountByType,
    TrafficSignalInstance,
    LaneMarkingResult,
    PerceptualChallengeResult,
    SafetyAssessmentResult,
    # Keys
    STAGE1_PROMPT,
    KEY_PROMPTS,
    KEY_SCHEMAS,
    KEY_EXTRACTORS,
    KEY_CATEGORIES,
    KEY_DIFFICULTY,
    DIFFICULTY_TO_TIER,
    KEYS,
    KEYS_ORIGINAL,
    KEYS_EXTENDED,
    get_prompt,
    get_schema,
    extract_value,
    get_keys_by_category,
    get_keys_by_difficulty,
    get_tier_for_key,
)

# For backward compatibility with bash script --export-json
import json


def export_json():
    """Export all key definitions as JSON (for bash script compatibility)."""
    from vlm.config import get_model_tiers, DEFAULT_KEY_TIERS

    tiers = get_model_tiers()

    return {
        "model_tiers": tiers,
        "stage1_prompt": STAGE1_PROMPT,
        "categories": KEY_CATEGORIES,
        "difficulty": KEY_DIFFICULTY,
        "keys": {
            key: {
                "prompt": KEY_PROMPTS[key],
                "schema": KEY_SCHEMAS[key],
                "tier": DEFAULT_KEY_TIERS.get(key, "medium"),
                "model": tiers[DEFAULT_KEY_TIERS.get(key, "medium")],
                "category": next((c for c, ks in KEY_CATEGORIES.items() if key in ks), "unknown"),
                "difficulty": KEY_DIFFICULTY.get(key, "medium"),
            }
            for key in KEYS
        },
    }


def export_env():
    """Export configuration as shell environment variables."""
    from vlm.config import get_model_tiers, DEFAULT_KEY_TIERS

    lines = ["# Scene Classification Configuration (generated)", ""]

    # Model tiers
    tiers = get_model_tiers()
    lines.append("# Model tiers")
    lines.append(f'MODEL_SMALL="{tiers["small"]}"')
    lines.append(f'MODEL_MEDIUM="{tiers["medium"]}"')
    lines.append(f'MODEL_LARGE="{tiers["large"]}"')
    lines.append("")

    # Key tiers
    lines.append("# Key-to-tier mappings")
    for key in KEYS:
        tier = DEFAULT_KEY_TIERS.get(key, "medium")
        lines.append(f'KEY_{key.upper()}_TIER="{tier}"')

    return "\n".join(lines)


# CLI for bash script compatibility
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Scene classification keys (wrapper)")
    parser.add_argument("--export-json", action="store_true", help="Export as JSON")
    parser.add_argument("--export-env", action="store_true", help="Export as shell env vars")
    parser.add_argument("--list-keys", action="store_true", help="List all keys")
    parser.add_argument("--list-extended", action="store_true", help="List extended keys")
    parser.add_argument("--list-by-category", metavar="CAT", help="List keys in category")
    parser.add_argument("--list-by-difficulty", metavar="DIFF", help="List keys by difficulty")
    parser.add_argument("--show-prompt", metavar="KEY", help="Show prompt for a key")
    parser.add_argument("--show-schema", metavar="KEY", help="Show schema for a key")
    parser.add_argument("--show-categories", action="store_true", help="Show all categories")
    args = parser.parse_args()

    if args.export_json:
        print(json.dumps(export_json(), indent=2))
    elif args.export_env:
        print(export_env())
    elif args.list_keys:
        for key in KEYS:
            diff = KEY_DIFFICULTY.get(key, "?")
            print(f"{key} [{diff}]")
    elif args.list_extended:
        for key in KEYS_EXTENDED:
            diff = KEY_DIFFICULTY.get(key, "?")
            print(f"{key} [{diff}]")
    elif args.list_by_category:
        keys = get_keys_by_category(args.list_by_category)
        for key in keys:
            print(key)
    elif args.list_by_difficulty:
        keys = get_keys_by_difficulty(args.list_by_difficulty)
        for key in keys:
            print(key)
    elif args.show_prompt:
        if args.show_prompt == "stage1":
            print(STAGE1_PROMPT)
        elif args.show_prompt in KEY_PROMPTS:
            print(KEY_PROMPTS[args.show_prompt])
        else:
            print(f"Unknown key: {args.show_prompt}", file=sys.stderr)
            sys.exit(1)
    elif args.show_schema:
        if args.show_schema in KEY_SCHEMAS:
            print(json.dumps(KEY_SCHEMAS[args.show_schema], indent=2))
        else:
            print(f"Unknown key: {args.show_schema}", file=sys.stderr)
            sys.exit(1)
    elif args.show_categories:
        for cat, keys in KEY_CATEGORIES.items():
            print(f"\n{cat.upper()}:")
            for key in keys:
                diff = KEY_DIFFICULTY.get(key, "?")
                print(f"  - {key} [{diff}]")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
