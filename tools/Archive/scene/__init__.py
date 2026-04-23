"""
Scene Classification Package

Centralized definitions for autonomous vehicle scene classification.

Usage:
    from scene import get_prompt, get_schema, get_response_model, KEYS
    from scene.enums import Weather, RoadType, SafetyCriticality
    from scene.responses import WeatherResponse, RoadTypeResponse

Example:
    from scene import get_prompt, get_schema, get_response_model

    # Get prompt and schema for a key
    prompt = get_prompt("weather")
    schema = get_schema("weather")  # JSON schema from Pydantic model
    model = get_response_model("weather")  # WeatherResponse class

    # Parse LLM response directly to Pydantic model
    response = model.model_validate_json(llm_output)
"""

# Enums
from .enums import (
    # Scene Context
    RoadType,
    Weather,
    TimeOfDay,
    TrafficSituation,
    # Spatial Reasoning
    OcclusionLevel,
    DepthComplexity,
    SpatialQueryType,
    # Perceptual
    VisualDegradation,
    # Safety
    SafetyCriticality,
    VulnerableRoadUserType,
    RequiredAction,
    HazardUrgency,
    # Attribute Binding
    TrafficLightState,
    LaneMarkingType,
    Confidence,
)

# Response Models
from .responses import (
    RESPONSE_MODELS,
    get_response_model,
    get_schema,
    # Individual response models
    TrafficSituationResponse,
    RoadTypeResponse,
    WeatherResponse,
    TimeOfDayResponse,
    PedestriansPresentResponse,
    CyclistsPresentResponse,
    ConstructionActivityResponse,
    TrafficSignalsVisibleResponse,
    VehicleCountResponse,
    NotableElementsResponse,
    OcclusionLevelResponse,
    DepthComplexityResponse,
    NearestVehicleDistanceResponse,
    SpatialRelationsResponse,
    VisualDegradationResponse,
    SimilarObjectConfusionResponse,
    EdgeCaseObjectsResponse,
    SafetyCriticalityResponse,
    VulnerableRoadUsersResponse,
    ImmediateHazardsResponse,
    RequiredActionResponse,
    PedestrianCountResponse,
    VehicleCountByTypeResponse,
    TrafficLightStatesResponse,
    LaneMarkingTypeResponse,
)

# Keys API
from .keys import (
    # Stage 1
    STAGE1_PROMPT,
    # Key data
    KEY_PROMPTS,
    KEY_CATEGORIES,
    KEY_DIFFICULTY,
    DIFFICULTY_TO_TIER,
    # Key lists
    KEYS,
    KEYS_ORIGINAL,
    KEYS_EXTENDED,
    # Functions
    get_prompt,
    get_keys_by_category,
    get_keys_by_difficulty,
    get_tier_for_key,
)

__all__ = [
    # Enums
    "RoadType",
    "Weather",
    "TimeOfDay",
    "TrafficSituation",
    "OcclusionLevel",
    "DepthComplexity",
    "SpatialQueryType",
    "VisualDegradation",
    "SafetyCriticality",
    "VulnerableRoadUserType",
    "RequiredAction",
    "HazardUrgency",
    "TrafficLightState",
    "LaneMarkingType",
    "Confidence",
    # Response Models
    "RESPONSE_MODELS",
    "get_response_model",
    "get_schema",
    "TrafficSituationResponse",
    "RoadTypeResponse",
    "WeatherResponse",
    "TimeOfDayResponse",
    "PedestriansPresentResponse",
    "CyclistsPresentResponse",
    "ConstructionActivityResponse",
    "TrafficSignalsVisibleResponse",
    "VehicleCountResponse",
    "NotableElementsResponse",
    "OcclusionLevelResponse",
    "DepthComplexityResponse",
    "NearestVehicleDistanceResponse",
    "SpatialRelationsResponse",
    "VisualDegradationResponse",
    "SimilarObjectConfusionResponse",
    "EdgeCaseObjectsResponse",
    "SafetyCriticalityResponse",
    "VulnerableRoadUsersResponse",
    "ImmediateHazardsResponse",
    "RequiredActionResponse",
    "PedestrianCountResponse",
    "VehicleCountByTypeResponse",
    "TrafficLightStatesResponse",
    "LaneMarkingTypeResponse",
    # Keys
    "STAGE1_PROMPT",
    "KEY_PROMPTS",
    "KEY_CATEGORIES",
    "KEY_DIFFICULTY",
    "DIFFICULTY_TO_TIER",
    "KEYS",
    "KEYS_ORIGINAL",
    "KEYS_EXTENDED",
    "get_prompt",
    "get_keys_by_category",
    "get_keys_by_difficulty",
    "get_tier_for_key",
]
