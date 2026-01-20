#!/bin/bash
#
# Test per-key classification prompts with Ollama
#
# Configuration is loaded from Python (scene_keys.py) for consistency.
# Override with environment variables if needed.
#
# Usage:
#   ./test_perkey.sh                    # Run all tests
#   ./test_perkey.sh reasoning          # Test Stage 1 only
#   ./test_perkey.sh traffic            # Test traffic_situation only
#   ./test_perkey.sh weather            # Test weather only
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# =============================================================================
# LOAD CONFIG FROM PYTHON (single source of truth)
# =============================================================================

# Source model tier config from Python if not already set
if [[ -z "${MODEL_SMALL:-}" ]]; then
    eval "$(python "${SCRIPT_DIR}/scene_keys.py" --export-env 2>/dev/null || echo '
MODEL_SMALL="qwen3-vl:4b"
MODEL_MEDIUM="qwen3-vl:8b"
MODEL_LARGE="qwen3-vl:30b"
')"
fi

# Endpoint configuration (can override with env vars)
OLLAMA_HOST="${OLLAMA_HOST:-${OLLAMA_URL:-http://localhost:11434}}"
OLLAMA_URL="${OLLAMA_HOST}/api/chat"

# =============================================================================
# LOAD PROMPTS AND SCHEMAS FROM PYTHON
# =============================================================================

# Cache the JSON export for this session
KEYS_JSON=""
get_keys_json() {
    if [[ -z "$KEYS_JSON" ]]; then
        KEYS_JSON=$(python "${SCRIPT_DIR}/scene_keys.py" --export-json 2>/dev/null || echo '{}')
    fi
    echo "$KEYS_JSON"
}

get_prompt() {
    local key="$1"
    if [[ "$key" == "stage1" ]]; then
        get_keys_json | jq -r '.stage1_prompt'
    else
        get_keys_json | jq -r ".keys[\"$key\"].prompt"
    fi
}

get_schema() {
    local key="$1"
    get_keys_json | jq -c ".keys[\"$key\"].schema"
}

get_model() {
    local key="$1"
    get_keys_json | jq -r ".keys[\"$key\"].model"
}

# =============================================================================
# TEST IMAGE
# =============================================================================

TEST_IMAGE="${PROJECT_ROOT}/experiments/test_scene_composite.jpg"

if [[ ! -f "$TEST_IMAGE" ]]; then
    # Fallback to first available classification image
    TEST_IMAGE=$(find "${PROJECT_ROOT}/experiments/classification_results/images" -name "*.jpg" 2>/dev/null | head -1 || echo "")
fi

if [[ -z "$TEST_IMAGE" || ! -f "$TEST_IMAGE" ]]; then
    echo "Warning: No test image found. Stage 1 tests will fail."
fi

# =============================================================================
# TEST FUNCTIONS
# =============================================================================

test_stage1_reasoning() {
    echo "=== Test: Stage 1 - Scene Reasoning (${MODEL_LARGE}) ==="
    echo "Image: ${TEST_IMAGE}"
    echo ""

    if [[ ! -f "$TEST_IMAGE" ]]; then
        echo "ERROR: Test image not found"
        return 1
    fi

    local img_b64=$(base64 -i "$TEST_IMAGE")
    local prompt=$(get_prompt "stage1")

    curl -s -X POST "$OLLAMA_URL" \
        -H "Content-Type: application/json" \
        -d @- <<EOF | jq -r '.message.content'
{
    "model": "${MODEL_LARGE}",
    "messages": [
        {"role": "system", "content": $(echo "$prompt" | jq -Rs .)},
        {"role": "user", "content": "Describe what you see in detail.", "images": ["${img_b64}"]}
    ],
    "stream": false,
    "options": {"num_ctx": 32768}
}
EOF
}

test_key() {
    local key="$1"
    local reasoning="${2:-}"
    local model=$(get_model "$key")
    local prompt=$(get_prompt "$key")
    local schema=$(get_schema "$key")

    # Use default reasoning if not provided
    if [[ -z "$reasoning" ]]; then
        reasoning="Urban street with several vehicles visible. Clear weather, daytime. A pedestrian is crossing at the crosswalk. Traffic light visible showing green."
    fi

    echo "=== Test: ${key} (${model}) ==="
    echo ""

    curl -s -X POST "$OLLAMA_URL" \
        -H "Content-Type: application/json" \
        -d @- <<EOF | jq '.message.content | fromjson'
{
    "model": "${model}",
    "messages": [
        {"role": "system", "content": $(echo "$prompt" | jq -Rs .)},
        {"role": "user", "content": "SCENE: ${reasoning}"}
    ],
    "format": ${schema},
    "stream": false,
    "options": {"num_ctx": 32768}
}
EOF
}

run_all_tests() {
    echo "Running all per-key classification tests..."
    echo "============================================="
    echo ""
    echo "Model tiers:"
    echo "  small:  ${MODEL_SMALL}"
    echo "  medium: ${MODEL_MEDIUM}"
    echo "  large:  ${MODEL_LARGE}"
    echo ""
    echo "Endpoint: ${OLLAMA_HOST}"
    echo ""

    # Stage 1
    REASONING=$(test_stage1_reasoning)
    echo "$REASONING"
    echo ""
    echo "---"
    echo ""

    # Stage 2 tests using the generated reasoning
    local keys=$(get_keys_json | jq -r '.keys | keys[]')
    for key in $keys; do
        test_key "$key" "$REASONING"
        echo ""
        echo "---"
        echo ""
    done

    echo "============================================="
    echo "All tests complete."
}

# =============================================================================
# MAIN
# =============================================================================

show_help() {
    echo "Usage: $0 [command] [reasoning]"
    echo ""
    echo "Commands:"
    echo "  all           Run all tests (default)"
    echo "  reasoning     Test Stage 1 scene reasoning"
    echo "  <key>         Test a specific key (e.g., weather, traffic_situation)"
    echo ""
    echo "Keys:"
    get_keys_json | jq -r '.keys | keys[]' | sed 's/^/  /'
    echo ""
    echo "Environment variables:"
    echo "  OLLAMA_HOST   Ollama server URL (default: http://localhost:11434)"
    echo "  MODEL_SMALL   Small tier model (default: ${MODEL_SMALL})"
    echo "  MODEL_MEDIUM  Medium tier model (default: ${MODEL_MEDIUM})"
    echo "  MODEL_LARGE   Large tier model (default: ${MODEL_LARGE})"
}

case "${1:-all}" in
    help|--help|-h)
        show_help
        ;;
    reasoning|stage1)
        test_stage1_reasoning
        ;;
    all)
        run_all_tests
        ;;
    *)
        # Assume it's a key name
        if get_keys_json | jq -e ".keys[\"$1\"]" > /dev/null 2>&1; then
            test_key "$1" "${2:-}"
        else
            echo "Unknown command or key: $1"
            echo ""
            show_help
            exit 1
        fi
        ;;
esac
