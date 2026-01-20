"""
Scene Classification Enums

All enum definitions for scene classification keys.
Organized by category for clarity.
"""

from enum import Enum


# =============================================================================
# SCENE CONTEXT ENUMS
# =============================================================================

class RoadType(str, Enum):
    """Road environment classification."""
    highway = "highway"
    urban_street = "urban_street"
    residential = "residential"
    intersection = "intersection"
    parking_lot = "parking_lot"
    construction_zone = "construction_zone"
    rural = "rural"


class Weather(str, Enum):
    """Weather condition classification."""
    clear = "clear"
    cloudy = "cloudy"
    rainy = "rainy"
    foggy = "foggy"
    snowy = "snowy"


class TimeOfDay(str, Enum):
    """Time of day classification."""
    day = "day"
    dawn_dusk = "dawn_dusk"
    night = "night"


class TrafficSituation(str, Enum):
    """Traffic complexity classification."""
    simple = "simple"
    moderate = "moderate"
    complex = "complex"
    critical = "critical"


# =============================================================================
# SPATIAL REASONING ENUMS
# =============================================================================

class OcclusionLevel(str, Enum):
    """Degree of object occlusion in scene."""
    none = "none"              # All objects fully visible
    minimal = "minimal"        # <25% of key objects partially hidden
    moderate = "moderate"      # 25-50% partial occlusion
    severe = "severe"          # >50% or critical objects occluded


class DepthComplexity(str, Enum):
    """Spatial depth layering complexity."""
    flat = "flat"              # Single depth plane (e.g., empty highway)
    layered = "layered"        # 2-3 distinct depth zones
    complex = "complex"        # Multiple overlapping depth planes


class SpatialQueryType(str, Enum):
    """Type of spatial reasoning required."""
    none = "none"
    relative_position = "relative_position"    # left/right/front/behind
    distance_estimation = "distance_estimation"
    size_comparison = "size_comparison"
    trajectory_prediction = "trajectory_prediction"
    occlusion_reasoning = "occlusion_reasoning"


# =============================================================================
# PERCEPTUAL CHALLENGE ENUMS
# =============================================================================

class VisualDegradation(str, Enum):
    """Primary visual quality degradation factor."""
    none = "none"
    glare = "glare"                    # Sun glare, reflections
    low_light = "low_light"            # Underexposure, dark areas
    motion_blur = "motion_blur"
    rain_artifacts = "rain_artifacts"  # Droplets, wet reflections
    fog_haze = "fog_haze"
    sensor_artifact = "sensor_artifact"  # Lens flare, dirt, etc.


# =============================================================================
# SAFETY CRITICAL ENUMS
# =============================================================================

class SafetyCriticality(str, Enum):
    """Failure consequence tier."""
    tier1_catastrophic = "tier1_catastrophic"  # Pedestrian/cyclist miss -> fatality
    tier2_severe = "tier2_severe"              # Lane/signal error -> collision
    tier3_moderate = "tier3_moderate"          # Speed/distance error -> minor incident
    tier4_minor = "tier4_minor"                # Aesthetic/comfort only


class VulnerableRoadUserType(str, Enum):
    """Categories of vulnerable road users (VRUs)."""
    none = "none"
    pedestrian_adult = "pedestrian_adult"
    pedestrian_child = "pedestrian_child"
    cyclist = "cyclist"
    motorcyclist = "motorcyclist"
    wheelchair_user = "wheelchair_user"
    construction_worker = "construction_worker"
    animal = "animal"


class RequiredAction(str, Enum):
    """Required driving action based on scene."""
    none = "none"      # Maintain current speed and course
    slow = "slow"      # Reduce speed, prepare to stop
    stop = "stop"      # Come to complete stop required
    evade = "evade"    # Steering maneuver may be needed


class HazardUrgency(str, Enum):
    """Urgency level of detected hazard."""
    immediate = "immediate"    # Requires instant response
    approaching = "approaching"  # Developing situation
    potential = "potential"    # Could become hazardous


# =============================================================================
# ATTRIBUTE BINDING ENUMS
# =============================================================================

class TrafficLightState(str, Enum):
    """Traffic signal state."""
    red = "red"
    yellow = "yellow"
    green = "green"
    flashing_red = "flashing_red"
    flashing_yellow = "flashing_yellow"
    off = "off"
    unknown = "unknown"


class LaneMarkingType(str, Enum):
    """Lane marking classification."""
    solid_white = "solid_white"
    dashed_white = "dashed_white"
    solid_yellow = "solid_yellow"
    dashed_yellow = "dashed_yellow"
    double_yellow = "double_yellow"
    none = "none"
    unknown = "unknown"


class Confidence(str, Enum):
    """Confidence level for classifications."""
    high = "high"
    medium = "medium"
    low = "low"
