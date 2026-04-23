"""
Scene Classification Response Models

Pydantic models for LLM responses. Each model corresponds to a classification key.
Use model.model_json_schema() to get the JSON schema for structured output.
Use model.model_validate_json(response) to parse LLM output directly.
"""

from pydantic import BaseModel, Field

from .enums import (
    RoadType,
    Weather,
    TimeOfDay,
    TrafficSituation,
    OcclusionLevel,
    DepthComplexity,
    VisualDegradation,
    SafetyCriticality,
    VulnerableRoadUserType,
    RequiredAction,
    TrafficLightState,
    LaneMarkingType,
    Confidence,
    HazardUrgency,
)


# =============================================================================
# SCENE CONTEXT RESPONSES
# =============================================================================

class TrafficSituationPoints(BaseModel):
    """Breakdown of traffic situation scoring."""
    vehicles: int = Field(ge=0, le=3)
    pedestrians: int = Field(ge=0, le=2)
    construction: int = Field(ge=0, le=3)
    intersection: int = Field(ge=0, le=2)
    signals: int = Field(ge=0, le=1)
    weather: int = Field(ge=0, le=2)
    visibility: int = Field(ge=0, le=1)


class TrafficSituationResponse(BaseModel):
    """Response for traffic_situation key."""
    points: TrafficSituationPoints
    total: int
    category: TrafficSituation


class RoadTypeResponse(BaseModel):
    """Response for road_type key."""
    reasoning: str
    road_type: RoadType


class WeatherResponse(BaseModel):
    """Response for weather key."""
    reasoning: str
    weather: Weather


class TimeOfDayResponse(BaseModel):
    """Response for time_of_day key."""
    reasoning: str
    time_of_day: TimeOfDay


# =============================================================================
# OBJECT DETECTION RESPONSES
# =============================================================================

class PedestriansPresentResponse(BaseModel):
    """Response for pedestrians_present key."""
    reasoning: str
    pedestrians_present: bool


class CyclistsPresentResponse(BaseModel):
    """Response for cyclists_present key."""
    reasoning: str
    cyclists_present: bool


class ConstructionActivityResponse(BaseModel):
    """Response for construction_activity key."""
    reasoning: str
    construction_activity: bool


class TrafficSignalsVisibleResponse(BaseModel):
    """Response for traffic_signals_visible key."""
    reasoning: str
    traffic_signals_visible: bool


class VehicleCountResponse(BaseModel):
    """Response for vehicle_count key."""
    reasoning: str
    vehicle_count: int = Field(ge=0, le=50)


class NotableElementsResponse(BaseModel):
    """Response for notable_elements key."""
    notable_elements: list[str]


# =============================================================================
# SPATIAL REASONING RESPONSES
# =============================================================================

class OcclusionLevelResponse(BaseModel):
    """Response for occlusion_level key."""
    reasoning: str
    occlusion_level: OcclusionLevel
    occluded_objects: list[str] = Field(default_factory=list)


class DepthComplexityResponse(BaseModel):
    """Response for depth_complexity key."""
    reasoning: str
    depth_complexity: DepthComplexity
    depth_zones: int = Field(ge=1, le=5, default=1)


class NearestVehicleDistanceResponse(BaseModel):
    """Response for nearest_vehicle_distance key."""
    reasoning: str
    vehicle_type: str
    estimated_meters: float = Field(ge=0)
    confidence: Confidence


class SpatialRelation(BaseModel):
    """A spatial relationship between two objects."""
    object_a: str
    object_b: str
    relation: str
    confidence: Confidence


class SpatialRelationsResponse(BaseModel):
    """Response for spatial_relations key."""
    relations: list[SpatialRelation] = Field(max_length=5)


# =============================================================================
# PERCEPTUAL CHALLENGE RESPONSES
# =============================================================================

class VisualDegradationResponse(BaseModel):
    """Response for visual_degradation key."""
    reasoning: str
    visual_degradation: VisualDegradation
    severity: str = Field(default="none", pattern=r"^(none|mild|moderate|severe)$")


class SimilarObjectConfusionResponse(BaseModel):
    """Response for similar_object_confusion key."""
    reasoning: str
    similar_object_confusion: bool
    confusion_candidates: list[str] = Field(default_factory=list)


class EdgeCaseObjectsResponse(BaseModel):
    """Response for edge_case_objects key."""
    edge_case_objects: list[str]
    risk_assessment: str = Field(default="")


# =============================================================================
# SAFETY CRITICAL RESPONSES
# =============================================================================

class SafetyCriticalityResponse(BaseModel):
    """Response for safety_criticality key."""
    reasoning: str
    safety_criticality: SafetyCriticality
    primary_risk: str


class VRUInstance(BaseModel):
    """Individual vulnerable road user detection."""
    type: VulnerableRoadUserType
    location: str
    occluded: bool
    in_path: bool


class VulnerableRoadUsersResponse(BaseModel):
    """Response for vulnerable_road_users key."""
    vrus: list[VRUInstance]
    total_count: int = Field(ge=0)


class HazardInstance(BaseModel):
    """Individual hazard detection."""
    description: str
    urgency: HazardUrgency


class ImmediateHazardsResponse(BaseModel):
    """Response for immediate_hazards key."""
    hazards: list[HazardInstance]


class RequiredActionResponse(BaseModel):
    """Response for required_action key."""
    reasoning: str
    required_action: RequiredAction


# =============================================================================
# COUNTING RESPONSES
# =============================================================================

class PedestrianCountResponse(BaseModel):
    """Response for pedestrian_count key."""
    reasoning: str
    count: int = Field(ge=0)
    confidence: Confidence
    occluded_estimate: int = Field(ge=0, default=0)


class VehicleCountByTypeResponse(BaseModel):
    """Response for vehicle_count_by_type key."""
    reasoning: str
    cars: int = Field(ge=0)
    suvs_trucks: int = Field(ge=0)
    commercial: int = Field(ge=0)
    motorcycles: int = Field(ge=0)
    other: int = Field(ge=0)
    confidence: Confidence

    @property
    def total(self) -> int:
        return self.cars + self.suvs_trucks + self.commercial + self.motorcycles + self.other


# =============================================================================
# ATTRIBUTE BINDING RESPONSES
# =============================================================================

class TrafficSignalInstance(BaseModel):
    """Individual traffic signal detection."""
    location: str
    state: TrafficLightState
    applicable_to_ego: bool


class TrafficLightStatesResponse(BaseModel):
    """Response for traffic_light_states key."""
    signals: list[TrafficSignalInstance]


class LaneMarkingTypeResponse(BaseModel):
    """Response for lane_marking_type key."""
    reasoning: str
    left_marking: LaneMarkingType
    right_marking: LaneMarkingType
    special_markings: list[str] = Field(default_factory=list)


# =============================================================================
# RESPONSE MODEL REGISTRY
# =============================================================================

RESPONSE_MODELS: dict[str, type[BaseModel]] = {
    # Scene Context
    "traffic_situation": TrafficSituationResponse,
    "road_type": RoadTypeResponse,
    "weather": WeatherResponse,
    "time_of_day": TimeOfDayResponse,
    # Object Detection
    "pedestrians_present": PedestriansPresentResponse,
    "cyclists_present": CyclistsPresentResponse,
    "construction_activity": ConstructionActivityResponse,
    "traffic_signals_visible": TrafficSignalsVisibleResponse,
    "vehicle_count": VehicleCountResponse,
    "notable_elements": NotableElementsResponse,
    # Spatial Reasoning
    "occlusion_level": OcclusionLevelResponse,
    "depth_complexity": DepthComplexityResponse,
    "nearest_vehicle_distance": NearestVehicleDistanceResponse,
    "spatial_relations": SpatialRelationsResponse,
    # Perceptual Challenges
    "visual_degradation": VisualDegradationResponse,
    "similar_object_confusion": SimilarObjectConfusionResponse,
    "edge_case_objects": EdgeCaseObjectsResponse,
    # Safety Critical
    "safety_criticality": SafetyCriticalityResponse,
    "vulnerable_road_users": VulnerableRoadUsersResponse,
    "immediate_hazards": ImmediateHazardsResponse,
    "required_action": RequiredActionResponse,
    # Counting
    "pedestrian_count": PedestrianCountResponse,
    "vehicle_count_by_type": VehicleCountByTypeResponse,
    # Attribute Binding
    "traffic_light_states": TrafficLightStatesResponse,
    "lane_marking_type": LaneMarkingTypeResponse,
}


def get_response_model(key: str) -> type[BaseModel]:
    """Get the response model class for a key."""
    return RESPONSE_MODELS[key]


def get_schema(key: str) -> dict:
    """Get the JSON schema for a key (from its response model)."""
    return RESPONSE_MODELS[key].model_json_schema()
