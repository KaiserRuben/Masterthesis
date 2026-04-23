"""
VLM Configuration

Centralized configuration for VLM inference with:
- Model tier definitions (single source of truth)
- Per-endpoint configuration
- Environment variable overrides
- YAML configuration files
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .types import ModelTier


# =============================================================================
# MODEL TIERS - Single source of truth for model assignments
# =============================================================================

DEFAULT_MODEL_TIERS: dict[str, str] = {
    "small": "qwen3-vl:4b",
    "medium": "qwen3-vl:8b",
    "large": "qwen3-vl:30b",
}


def get_model_tiers(config: "VLMConfig | None" = None) -> dict[str, str]:
    """
    Get model tier definitions.

    Priority (highest to lowest):
        1. Environment variables (MODEL_SMALL, MODEL_MEDIUM, MODEL_LARGE)
        2. Config's model_tiers (if config provided)
        3. DEFAULT_MODEL_TIERS

    Args:
        config: Optional VLMConfig to use as base (before env overrides)
    """
    # Start with defaults or config
    if config is not None:
        base = config.model_tiers.copy()
    else:
        base = DEFAULT_MODEL_TIERS.copy()

    # Apply env overrides
    if model := os.environ.get("MODEL_SMALL"):
        base["small"] = model
    if model := os.environ.get("MODEL_MEDIUM"):
        base["medium"] = model
    if model := os.environ.get("MODEL_LARGE"):
        base["large"] = model

    return base


def resolve_tier(tier_or_model: str, config: "VLMConfig | None" = None) -> str:
    """
    Resolve a tier name to a model name, or return as-is if already a model.

    Args:
        tier_or_model: Either "small"/"medium"/"large" or a model name like "qwen3-vl:8b"
        config: Optional VLMConfig to use for tier resolution

    Returns:
        Model name
    """
    tiers = get_model_tiers(config)
    return tiers.get(tier_or_model, tier_or_model)


# =============================================================================
# DEFAULT KEY-TO-TIER MAPPINGS
# =============================================================================

DEFAULT_KEY_TIERS: dict[str, str] = {
    # Stage 1 reasoning (always large)
    "stage1": "large",

    # --- Scene Context ---
    "weather": "small",
    "time_of_day": "small",
    "road_type": "small",
    "traffic_situation": "medium",

    # --- Object Detection ---
    "pedestrians_present": "small",
    "cyclists_present": "small",
    "construction_activity": "small",
    "traffic_signals_visible": "medium",
    "vehicle_count": "medium",
    "notable_elements": "medium",

    # --- Spatial Reasoning ---
    "occlusion_level": "medium",
    "depth_complexity": "large",
    "nearest_vehicle_distance": "large",
    "spatial_relations": "large",

    # --- Perceptual Challenges ---
    "visual_degradation": "medium",
    "similar_object_confusion": "large",
    "edge_case_objects": "large",

    # --- Safety Critical ---
    "safety_criticality": "large",
    "vulnerable_road_users": "large",
    "immediate_hazards": "large",
    "required_action": "large",

    # --- Counting & Quantification ---
    "pedestrian_count": "large",
    "vehicle_count_by_type": "large",

    # --- Attribute Binding ---
    "traffic_light_states": "large",
    "lane_marking_type": "large",
}


# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================

@dataclass
class EndpointConfig:
    """Configuration for a single Ollama endpoint."""
    url: str = "http://localhost:11434"
    max_concurrent: int = 1
    timeout_seconds: int = 900
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EndpointConfig":
        return cls(
            url=data.get("url", "http://localhost:11434"),
            max_concurrent=data.get("max_concurrent", 1),
            timeout_seconds=data.get("timeout_seconds", 900),
            retry_attempts=data.get("retry_attempts", 3),
            retry_delay_seconds=data.get("retry_delay_seconds", 1.0),
        )


@dataclass
class ModelEndpointConfig:
    """Configuration mapping a model to endpoint(s)."""
    model: str
    endpoints: list[str] = field(default_factory=lambda: ["default"])
    tier: ModelTier = ModelTier.MEDIUM
    context_window: int = 32768*2

    @property
    def endpoint(self) -> str:
        """Primary endpoint (backward compatibility)."""
        return self.endpoints[0] if self.endpoints else "default"

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> "ModelEndpointConfig":
        tier_str = data.get("tier", "medium")

        # Handle both single endpoint and endpoints list
        if "endpoints" in data:
            endpoints = data["endpoints"]
            if isinstance(endpoints, str):
                endpoints = [endpoints]
        elif "endpoint" in data:
            endpoints = [data["endpoint"]]
        else:
            endpoints = ["default"]

        return cls(
            model=name,
            endpoints=endpoints,
            tier=ModelTier(tier_str),
            context_window=data.get("context_window", 32768),
        )


@dataclass
class KeyModelMapping:
    """
    Maps classification keys to models/tiers.

    Values can be:
    - Tier names: "small", "medium", "large"
    - Model names: "qwen3-vl:8b" (used directly)

    Note: For proper config-based tier resolution, use VLMConfig.get_model_for_key()
    which uses the config's model_tiers. Direct use of KeyModelMapping.get() falls
    back to environment variable based resolution.
    """
    _mappings: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        # Initialize with defaults if empty
        if not self._mappings:
            self._mappings = DEFAULT_KEY_TIERS.copy()

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> "KeyModelMapping":
        mappings = DEFAULT_KEY_TIERS.copy()
        mappings.update(data)
        return cls(_mappings=mappings)

    def get(self, key: str) -> str:
        """Get the resolved model name for a key."""
        tier_or_model = self._mappings.get(key, "medium")
        return resolve_tier(tier_or_model)

    def get_tier(self, key: str) -> str:
        """Get the raw tier/model value for a key (before resolution)."""
        return self._mappings.get(key, "medium")

    def set(self, key: str, tier_or_model: str):
        """Set the tier/model for a key."""
        self._mappings[key] = tier_or_model

    def keys(self) -> list[str]:
        """Get all configured keys."""
        return list(self._mappings.keys())

    def to_dict(self) -> dict[str, str]:
        """Export as dictionary (resolved model names)."""
        return {k: self.get(k) for k in self._mappings}


@dataclass
class VLMConfig:
    """Complete VLM configuration."""
    endpoints: dict[str, EndpointConfig] = field(default_factory=dict)
    models: dict[str, ModelEndpointConfig] = field(default_factory=dict)
    key_mapping: KeyModelMapping = field(default_factory=KeyModelMapping)
    model_tiers: dict[str, str] = field(default_factory=lambda: DEFAULT_MODEL_TIERS.copy())

    def __post_init__(self):
        if "default" not in self.endpoints:
            self.endpoints["default"] = EndpointConfig()

    def get_endpoint_for_model(self, model_name: str) -> EndpointConfig:
        """Get the endpoint configuration for a model."""
        if model_name in self.models:
            endpoint_name = self.models[model_name].endpoint
            return self.endpoints.get(endpoint_name, self.endpoints["default"])
        return self.endpoints["default"]

    def get_model_config(self, model_name: str) -> ModelEndpointConfig:
        """Get model configuration, creating default if needed."""
        if model_name not in self.models:
            self.models[model_name] = ModelEndpointConfig(model=model_name)
        return self.models[model_name]

    def resolve_tier(self, tier_or_model: str) -> str:
        """
        Resolve a tier name to a model name using this config's model_tiers.

        Args:
            tier_or_model: Either "small"/"medium"/"large" or a model name like "qwen3-vl:8b"

        Returns:
            Model name
        """
        return self.model_tiers.get(tier_or_model, tier_or_model)

    def get_model_for_key(self, key: str) -> str:
        """Get the resolved model name for a classification key."""
        tier_or_model = self.key_mapping.get_tier(key)
        return self.resolve_tier(tier_or_model)

    def get_endpoints_for_model(self, model: str) -> list[str]:
        """Get all endpoints that support a model."""
        if model in self.models:
            return self.models[model].endpoints
        return ["default"]

    def get_supported_models(self, endpoint_name: str) -> set[str]:
        """Get all models supported by an endpoint."""
        return {
            name for name, cfg in self.models.items()
            if endpoint_name in cfg.endpoints
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VLMConfig":
        """Create config from dictionary."""
        endpoints = {
            name: EndpointConfig.from_dict(cfg)
            for name, cfg in data.get("endpoints", {}).items()
        }

        models = {
            name: ModelEndpointConfig.from_dict(name, cfg)
            for name, cfg in data.get("models", {}).items()
        }

        key_mapping = KeyModelMapping.from_dict(data.get("key_mapping", {}))

        # Parse model_tiers with defaults
        model_tiers = DEFAULT_MODEL_TIERS.copy()
        if "model_tiers" in data:
            model_tiers.update(data["model_tiers"])

        return cls(endpoints=endpoints, models=models, key_mapping=key_mapping, model_tiers=model_tiers)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "VLMConfig":
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            return cls()

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        return cls.from_dict(data)

    @classmethod
    def from_env(cls, base_config: "VLMConfig | None" = None) -> "VLMConfig":
        """
        Override configuration from environment variables.

        Supported env vars:
            OLLAMA_URL: Default endpoint URL
            OLLAMA_HOST: Alias for OLLAMA_URL

            MODEL_SMALL: Model for small tier (e.g., qwen3-vl:4b)
            MODEL_MEDIUM: Model for medium tier (e.g., qwen3-vl:8b)
            MODEL_LARGE: Model for large tier (e.g., qwen3-vl:30b)

            VLM_ENDPOINT_{NAME}_URL: Specific endpoint URL
            VLM_ENDPOINT_{NAME}_MAX_CONCURRENT: Concurrency limit

            VLM_KEY_{KEY}_MODEL: Model for specific key (e.g., VLM_KEY_STAGE1_MODEL=qwen3-vl:30b)
            VLM_KEY_{KEY}_TIER: Tier for specific key (e.g., VLM_KEY_WEATHER_TIER=small)
        """
        config = base_config or cls()

        # Default endpoint from OLLAMA_URL or OLLAMA_HOST
        if url := os.environ.get("OLLAMA_URL") or os.environ.get("OLLAMA_HOST"):
            config.endpoints["default"].url = url

        # Model tier overrides
        if model := os.environ.get("MODEL_SMALL"):
            config.model_tiers["small"] = model
        if model := os.environ.get("MODEL_MEDIUM"):
            config.model_tiers["medium"] = model
        if model := os.environ.get("MODEL_LARGE"):
            config.model_tiers["large"] = model

        # Per-endpoint overrides
        for key, value in os.environ.items():
            if key.startswith("VLM_ENDPOINT_") and key.endswith("_URL"):
                name = key[13:-4].lower()
                if name not in config.endpoints:
                    config.endpoints[name] = EndpointConfig()
                config.endpoints[name].url = value

            elif key.startswith("VLM_ENDPOINT_") and key.endswith("_MAX_CONCURRENT"):
                name = key[13:-15].lower()
                if name not in config.endpoints:
                    config.endpoints[name] = EndpointConfig()
                config.endpoints[name].max_concurrent = int(value)

        # Key-to-model/tier overrides
        for key, value in os.environ.items():
            if key.startswith("VLM_KEY_") and key.endswith("_MODEL"):
                # VLM_KEY_STAGE1_MODEL=qwen3-vl:30b -> stage1 = qwen3-vl:30b
                key_name = key[8:-6].lower()
                config.key_mapping.set(key_name, value)

            elif key.startswith("VLM_KEY_") and key.endswith("_TIER"):
                # VLM_KEY_WEATHER_TIER=small -> weather = small
                key_name = key[8:-5].lower()
                config.key_mapping.set(key_name, value.lower())

        return config

    def to_env_script(self) -> str:
        """Export configuration as shell environment variables."""
        lines = ["# VLM Configuration (generated)"]

        # Model tiers
        lines.append(f'export MODEL_SMALL="{self.model_tiers["small"]}"')
        lines.append(f'export MODEL_MEDIUM="{self.model_tiers["medium"]}"')
        lines.append(f'export MODEL_LARGE="{self.model_tiers["large"]}"')
        lines.append("")

        # Endpoints
        for name, endpoint in self.endpoints.items():
            prefix = f"VLM_ENDPOINT_{name.upper()}"
            lines.append(f'export {prefix}_URL="{endpoint.url}"')
            lines.append(f'export {prefix}_MAX_CONCURRENT="{endpoint.max_concurrent}"')
        lines.append("")

        # Key mappings
        for key in self.key_mapping.keys():
            tier = self.key_mapping.get_tier(key)
            lines.append(f'export VLM_KEY_{key.upper()}_TIER="{tier}"')

        return "\n".join(lines)


# =============================================================================
# LOADING
# =============================================================================

def load_config(
    config_path: Path | str | None = None,
    apply_env: bool = True,
) -> VLMConfig:
    """
    Load configuration from file and environment.

    Priority (highest to lowest):
    1. Environment variables
    2. Config file
    3. Defaults
    """
    # Try default config locations if not specified
    if config_path is None:
        search_paths = [
            Path.cwd() / "vlm_config.yaml",
            Path.cwd() / ".vlm_config.yaml",
            Path.home() / ".config" / "vlm" / "config.yaml",
        ]
        for path in search_paths:
            if path.exists():
                config_path = path
                break

    # Load from file or create default
    if config_path and Path(config_path).exists():
        config = VLMConfig.from_yaml(config_path)
    else:
        config = VLMConfig()

    # Apply environment overrides
    if apply_env:
        config = VLMConfig.from_env(config)

    return config


# =============================================================================
# CONFIG TEMPLATE
# =============================================================================

DEFAULT_CONFIG_TEMPLATE = """
# VLM Configuration
# Save as vlm_config.yaml in your project root

# Model tiers - maps tier names to actual model names
# Can be overridden via MODEL_SMALL, MODEL_MEDIUM, MODEL_LARGE env vars
model_tiers:
  small: qwen3-vl:4b
  medium: qwen3-vl:8b
  large: qwen3-vl:30b

endpoints:
  default:
    url: http://localhost:11434
    max_concurrent: 1
    timeout_seconds: 900
    retry_attempts: 3

  # Example: separate endpoint for work-stealing queue
  # gpu_server:
  #   url: http://gpu-server:11434
  #   max_concurrent: 2
  # lightweight:
  #   url: http://light:11434
  #   max_concurrent: 4

models:
  qwen3-vl:4b:
    endpoint: default          # Single endpoint (backward compatible)
    tier: small
    context_window: 32768

  qwen3-vl:8b:
    endpoint: default
    tier: medium
    context_window: 32768

  qwen3-vl:30b:
    endpoint: default
    tier: large
    context_window: 32768

  # Example: multi-endpoint configuration for work-stealing
  # qwen3-vl:4b:
  #   endpoints: [gpu_server, lightweight]  # Available on both
  #   tier: small
  # qwen3-vl:8b:
  #   endpoints: [gpu_server, lightweight]  # Available on both
  #   tier: medium
  # qwen3-vl:30b:
  #   endpoints: [gpu_server]               # Only on GPU server
  #   tier: large

# Key-to-tier mappings (use tier names, not model names)
# Can also use model names directly if needed
key_mapping:
  stage1: large

  # --- Scene Context (easy = small, medium = medium) ---
  weather: small
  time_of_day: small
  road_type: small
  traffic_situation: medium

  # --- Object Detection ---
  pedestrians_present: small
  cyclists_present: small
  construction_activity: small
  traffic_signals_visible: medium
  vehicle_count: medium
  notable_elements: medium

  # --- Spatial Reasoning (medium to large) ---
  occlusion_level: medium
  depth_complexity: large
  nearest_vehicle_distance: large
  spatial_relations: large

  # --- Perceptual Challenges ---
  visual_degradation: medium
  similar_object_confusion: large
  edge_case_objects: large

  # --- Safety Critical (always large) ---
  safety_criticality: large
  vulnerable_road_users: large
  immediate_hazards: large
  required_action: large

  # --- Counting & Quantification (hard = large) ---
  pedestrian_count: large
  vehicle_count_by_type: large

  # --- Attribute Binding (hard = large) ---
  traffic_light_states: large
  lane_marking_type: large
"""
