"""
Scene Classification Keys

Centralized definitions for scene classification:
- Prompts for each classification key
- Response models for structured output
- Key metadata (category, difficulty, tier)
"""

from typing import Any

from pydantic import BaseModel

from .responses import RESPONSE_MODELS, get_response_model, get_schema


# =============================================================================
# STAGE 1 PROMPT
# =============================================================================

STAGE1_PROMPT = """You are analyzing a 4-camera autonomous vehicle view.
Layout: Top-left=Left peripheral 120°, Top-right=Front wide 120°, Bottom-left=Right peripheral 120°, Bottom-right=Front telephoto 30°.

Provide a DETAILED description of the scene. For each camera view, note:
- Road characteristics and markings
- All vehicles (count and describe)
- Any people (pedestrians, workers, cyclists)
- Traffic infrastructure (signals, signs)
- Weather and lighting conditions
- Construction equipment or activity
- Any hazards or unusual elements

Be extremely thorough - your description will be used for multiple downstream classifications."""


# =============================================================================
# KEY PROMPTS
# =============================================================================

KEY_PROMPTS: dict[str, str] = {
    # --- Scene Context ---
    "traffic_situation": """Classify TRAFFIC SITUATION using additive scoring.

SCORING:
- Vehicles: 0 pts (0-3), +1 (4-8), +2 (9-15), +3 (16+)
- Pedestrians visible: +2 pts
- Construction activity: +3 pts
- Intersection: +2 pts
- Active traffic signals: +1 pt
- Adverse weather: +2 pts
- Low visibility (night/dusk): +1 pt

CATEGORIES:
- simple (0-2 pts): Free-flowing, minimal complexity
- moderate (3-5 pts): Some attention required
- complex (6-8 pts): High attention required
- critical (9+ pts): Maximum vigilance

Calculate each scoring component, sum the total, and determine the category.""",

    "road_type": """Classify ROAD TYPE based on scene description.

CRITERIA:
- highway: Multiple lanes, high speed, limited access, barriers between directions
- urban_street: City environment, buildings nearby, moderate speeds
- residential: Quiet neighborhood, houses visible, low speed limits
- intersection: Junction of roads, turn lanes, multiple directions
- parking_lot: Parking spaces, slow movement, pedestrian mix
- construction_zone: Active construction, workers, equipment, altered lanes
- rural: Open countryside, few buildings, natural surroundings""",

    "weather": """Classify WEATHER based on scene description.

CRITERIA:
- clear: Blue/bright sky, sharp shadows, dry surfaces
- cloudy: Overcast, diffuse lighting, no precipitation
- rainy: Wet surfaces, rain drops, wipers, reflections
- foggy: Reduced visibility, hazy, washed out colors
- snowy: Snow visible on ground or falling, winter conditions""",

    "time_of_day": """Classify TIME OF DAY based on scene description.

CRITERIA:
- day: Bright natural light, clear visibility, sun high
- dawn_dusk: Low sun angle, orange/pink sky, long shadows
- night: Artificial lighting dominant, dark sky, headlights visible""",

    # --- Object Detection ---
    "pedestrians_present": """Are PEDESTRIANS present in the scene?

Count as pedestrians:
- People walking on sidewalks/crosswalks
- Construction workers
- People waiting at bus stops
- Anyone on foot""",

    "cyclists_present": """Are CYCLISTS present in the scene?

Look for:
- Bicycles being ridden
- People on bikes in bike lanes
- Cyclists on road edges

Note: Parked/stationary bikes without riders = false""",

    "construction_activity": """Is there CONSTRUCTION ACTIVITY in the scene?

Signs of construction:
- Excavators, cranes, construction vehicles
- Workers in safety vests/hard hats
- Traffic cones, barriers, construction signs
- Road work, debris, altered lane markings""",

    "traffic_signals_visible": """Are TRAFFIC SIGNALS visible in the scene?

Include:
- Traffic lights (any color/state)
- Stop signs
- Yield signs
- Railroad crossing signals""",

    "vehicle_count": """Count VEHICLES visible in the scene.

Include:
- Cars, trucks, buses, motorcycles
- Parked and moving vehicles
- Vehicles in all camera views

Count each distinct vehicle once. Range: 0-50.""",

    "notable_elements": """List all NOTABLE ELEMENTS in the scene.

Extract from description:
- Unusual objects or hazards
- Emergency vehicles
- Road features (crosswalks, speed bumps)
- Signage details
- Any safety-relevant items""",

    # --- Spatial Reasoning ---
    "occlusion_level": """Assess OCCLUSION LEVEL in the scene.

Examine all camera views for objects that are partially hidden:
- Vehicles behind other vehicles
- Pedestrians partially behind obstacles
- Signs/signals blocked by trees, trucks, etc.

CATEGORIES:
- none: All relevant objects fully visible
- minimal: <25% of key objects have minor occlusion
- moderate: 25-50% of objects partially occluded
- severe: >50% occlusion OR safety-critical objects hidden

Focus on objects relevant to driving decisions.""",

    "depth_complexity": """Assess DEPTH COMPLEXITY of the scene.

Analyze the spatial layering:
- How many distinct depth planes contain relevant objects?
- Are there overlapping objects at different distances?

CATEGORIES:
- flat: Single depth plane (empty road, clear highway)
- layered: 2-3 distinct depth zones with objects
- complex: Multiple overlapping depth planes, objects at many distances

Consider: near-field (<10m), mid-field (10-50m), far-field (>50m).""",

    "nearest_vehicle_distance": """Estimate DISTANCE to the nearest vehicle.

Analyze the front camera views to estimate the distance in meters to the closest vehicle (moving or stationary) in the ego vehicle's lane or adjacent lanes.

Provide:
- The type of vehicle (car, truck, motorcycle, etc.)
- Estimated distance in meters
- Confidence level (high/medium/low)

If no vehicles visible, state "none" with high confidence.""",

    "spatial_relations": """Identify KEY SPATIAL RELATIONS between objects.

List up to 5 critical spatial relationships relevant to driving:
- Relative positions (left_of, right_of, in_front_of, behind)
- Distance comparisons (closer_than, farther_than)
- Path intersections (crossing_path, same_lane, adjacent_lane)

For each relation, specify:
- object_a and object_b
- the relation type
- confidence (high/medium/low)

Focus on safety-relevant relationships.""",

    # --- Perceptual Challenges ---
    "visual_degradation": """Identify PRIMARY VISUAL DEGRADATION factor.

Examine image quality issues:
- glare: Sun glare, bright reflections, overexposure
- low_light: Dark areas, underexposure, poor contrast
- motion_blur: Blurred moving objects
- rain_artifacts: Water droplets, wet surface reflections
- fog_haze: Reduced visibility, hazy appearance
- sensor_artifact: Lens flare, dirt on lens, compression artifacts
- none: No significant degradation

Select the MOST impactful degradation factor.""",

    "similar_object_confusion": """Is there potential for SIMILAR OBJECT CONFUSION?

Look for scenarios where multiple similar objects could be confused:
- Multiple vehicles of same color/type in close proximity
- Repeated patterns (row of parked cars, multiple pedestrians)
- Objects that look similar but require different responses

Answer true if a VLM might confuse or miscount similar objects.""",

    "edge_case_objects": """List any EDGE CASE OBJECTS in the scene.

Objects that are unusual or might confuse vision models:
- Animals (dogs, deer, birds)
- Debris or fallen cargo
- Unusual vehicles (tractors, oversized loads)
- Temporary structures (tents, road work equipment)
- Misleading visual elements (billboards with people, reflections)
- Objects in unexpected locations

These are objects outside typical training distributions.""",

    # --- Safety Critical ---
    "safety_criticality": """Assess SAFETY CRITICALITY tier of the scene.

TIERS (select highest applicable):
- tier1_catastrophic: Vulnerable road users in/near path - miss could cause fatality
- tier2_severe: Active intersection, lane merge, or signal compliance required - error causes collision
- tier3_moderate: Speed/distance judgment needed, parking maneuvers - error causes minor incident
- tier4_minor: Open road, no hazards - errors have minimal consequence

Consider: What is the worst outcome if a VLM misclassifies this scene?""",

    "vulnerable_road_users": """Detect all VULNERABLE ROAD USERS (VRUs).

For each VRU, identify:
- Type: pedestrian_adult, pedestrian_child, cyclist, motorcyclist, wheelchair_user, construction_worker, animal
- Location: crosswalk, sidewalk, road_edge, in_lane, bike_lane
- Occluded: true/false (partially hidden)
- In path: true/false (in or approaching ego vehicle's trajectory)

VRUs require the highest detection reliability. List all instances.""",

    "immediate_hazards": """List all IMMEDIATE HAZARDS requiring response.

Hazards that need driver/system attention within seconds:
- Pedestrians entering roadway
- Vehicles braking ahead
- Obstacles in lane
- Traffic signal changes
- Emergency vehicles
- Animals in road

For each hazard, briefly describe what it is and why it's hazardous.""",

    "required_action": """Determine REQUIRED ACTION for this scene.

Based on all visible hazards and traffic situation:
- none: Maintain current speed and course
- slow: Reduce speed, prepare to stop
- stop: Come to complete stop required
- evade: Steering maneuver may be needed

Select the most conservative action warranted by the scene.""",

    # --- Counting & Quantification ---
    "pedestrian_count": """Count PEDESTRIANS with confidence assessment.

Count all people on foot visible across all camera views:
- Include: walkers, workers, people at bus stops, joggers
- Exclude: people inside vehicles

Provide:
- Total count
- Confidence (high if clear view, medium if some occlusion, low if uncertain)
- Estimated additional occluded pedestrians (best guess of hidden people)""",

    "vehicle_count_by_type": """Count VEHICLES by type.

Categorize and count all visible vehicles:
- cars: Sedans, hatchbacks, coupes
- suvs_trucks: SUVs, pickups, vans
- commercial: Buses, delivery trucks, semis
- motorcycles: Motorcycles, scooters
- other: Bicycles, construction vehicles, emergency vehicles

For each category, provide count and confidence level.""",

    # --- Attribute Binding ---
    "traffic_light_states": """Identify all TRAFFIC LIGHT STATES.

For each visible traffic signal:
- Location (e.g., "ahead center", "left turn lane", "cross traffic")
- State: red, yellow, green, flashing, off/unknown
- Applicable to ego vehicle: true/false

This tests attribute binding - correctly associating state with specific signal.""",

    "lane_marking_type": """Classify LANE MARKINGS in ego vehicle's lane.

Identify the lane markings on both sides of the ego vehicle's current lane:
- Left side: solid_white, dashed_white, solid_yellow, dashed_yellow, double_yellow, none
- Right side: same options
- Special: merge, turn_only, bike_lane_adjacent, construction_altered

Lane markings determine legal maneuvers.""",
}


# =============================================================================
# KEY METADATA
# =============================================================================

KEY_CATEGORIES: dict[str, list[str]] = {
    "scene_context": [
        "road_type", "weather", "time_of_day", "traffic_situation",
    ],
    "object_detection": [
        "pedestrians_present", "cyclists_present", "construction_activity",
        "traffic_signals_visible", "vehicle_count", "notable_elements",
    ],
    "spatial_reasoning": [
        "occlusion_level", "depth_complexity", "nearest_vehicle_distance", "spatial_relations",
    ],
    "perceptual_challenge": [
        "visual_degradation", "similar_object_confusion", "edge_case_objects",
    ],
    "safety_critical": [
        "safety_criticality", "vulnerable_road_users", "immediate_hazards", "required_action",
    ],
    "counting_quantification": [
        "pedestrian_count", "vehicle_count_by_type",
    ],
    "attribute_binding": [
        "traffic_light_states", "lane_marking_type",
    ],
}


KEY_DIFFICULTY: dict[str, str] = {
    # Easy - minimal degradation expected with smaller models
    "weather": "easy",
    "time_of_day": "easy",
    "road_type": "easy",
    "pedestrians_present": "easy",
    "cyclists_present": "easy",
    "construction_activity": "easy",

    # Medium - some degradation expected
    "traffic_signals_visible": "medium",
    "vehicle_count": "medium",
    "traffic_situation": "medium",
    "visual_degradation": "medium",
    "notable_elements": "medium",
    "occlusion_level": "medium",

    # Hard - significant degradation expected
    "depth_complexity": "hard",
    "nearest_vehicle_distance": "hard",
    "spatial_relations": "hard",
    "similar_object_confusion": "hard",
    "pedestrian_count": "hard",
    "vehicle_count_by_type": "hard",
    "traffic_light_states": "hard",
    "lane_marking_type": "hard",
    "vulnerable_road_users": "hard",

    # Critical - most prone to failure
    "safety_criticality": "critical",
    "immediate_hazards": "critical",
    "required_action": "critical",
    "edge_case_objects": "critical",
}


# Map difficulty to model tier
DIFFICULTY_TO_TIER: dict[str, str] = {
    "easy": "small",
    "medium": "medium",
    "hard": "large",
    "critical": "large",
}


# =============================================================================
# KEY REGISTRY
# =============================================================================

KEYS: list[str] = list(KEY_PROMPTS.keys())

KEYS_ORIGINAL: list[str] = [
    "traffic_situation", "road_type", "weather", "time_of_day",
    "pedestrians_present", "cyclists_present", "construction_activity",
    "traffic_signals_visible", "vehicle_count", "notable_elements",
]

KEYS_EXTENDED: list[str] = [k for k in KEYS if k not in KEYS_ORIGINAL]


# =============================================================================
# PUBLIC API
# =============================================================================

def get_prompt(key: str) -> str:
    """Get the prompt for a key."""
    return KEY_PROMPTS[key]


def get_keys_by_category(category: str) -> list[str]:
    """Get all keys in a category."""
    return KEY_CATEGORIES.get(category, [])


def get_keys_by_difficulty(difficulty: str) -> list[str]:
    """Get all keys at a difficulty level."""
    return [k for k, d in KEY_DIFFICULTY.items() if d == difficulty]


def get_tier_for_key(key: str) -> str:
    """Get the recommended model tier for a key based on difficulty."""
    difficulty = KEY_DIFFICULTY.get(key, "medium")
    return DIFFICULTY_TO_TIER.get(difficulty, "medium")


# Re-export from responses for backwards compatibility
__all__ = [
    "STAGE1_PROMPT",
    "KEY_PROMPTS",
    "KEYS",
    "KEYS_ORIGINAL",
    "KEYS_EXTENDED",
    "get_prompt",
    "get_schema",
    "get_response_model",
    "get_keys_by_category",
    "get_keys_by_difficulty",
    "get_tier_for_key",
]
