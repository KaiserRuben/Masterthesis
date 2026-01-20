# =============================================================================
# Makefile for Docker Operations
# =============================================================================

.PHONY: help build test notebook shell classify clean logs

# Default target
help:
	@echo "Alpamayo-R1 Docker Commands"
	@echo "==========================="
	@echo ""
	@echo "Setup:"
	@echo "  make build       Build Docker image"
	@echo "  make setup       Copy .env.example to .env (edit with your HF_TOKEN)"
	@echo ""
	@echo "Run experiments:"
	@echo "  make test        Run basic inference test (5 scenes)"
	@echo "  make notebook    Start Jupyter notebook (port 8888)"
	@echo "  make shell       Interactive bash shell"
	@echo ""
	@echo "With arguments:"
	@echo "  make run ARGS='-n 10'                    # 10 random scenes"
	@echo "  make run ARGS='--clip-id abc123'         # Specific clip"
	@echo "  make run ARGS='--list-clips'             # List clips"
	@echo ""
	@echo "Access results:"
	@echo "  Results are saved to: experiments/workstation/inference_results/"
	@echo "  make results                             # List result files"
	@echo ""
	@echo "Maintenance:"
	@echo "  make logs        Show container logs"
	@echo "  make clean       Remove containers and images"
	@echo "  make clean-all   Remove everything including cached models"
	@echo ""

# Build the Docker image (builds only the builder service which has the Dockerfile)
build:
	docker compose build builder

# Setup environment file
setup:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env from .env.example"; \
		echo ">>> Edit .env and add your HF_TOKEN <<<"; \
	else \
		echo ".env already exists"; \
	fi

# Run basic inference test
test: build
	docker compose run --rm test

# Run with custom arguments
ARGS ?=
run: build
	docker compose run --rm test $(ARGS)

# Start Jupyter notebook
notebook: build
	docker compose up notebook

# Interactive shell
shell: build
	docker compose run --rm shell

# Run batch scene classification
classify: build
	docker compose run --rm classify $(ARGS)

# List result files
results:
	@echo "Results in experiments/workstation/inference_results/:"
	@ls -la ../experiments/workstation/inference_results/*.json 2>/dev/null || echo "  (no results yet)"

# Show latest result
latest:
	@echo "Latest result:"
	@ls -t ../experiments/workstation/inference_results/*.json 2>/dev/null | head -1 | xargs cat | head -50

# Show logs
logs:
	docker compose logs -f

# Clean containers
clean:
	docker compose down --remove-orphans
	docker rmi alpamayo-runner:latest 2>/dev/null || true

# Clean everything including cached models
clean-all: clean
	docker volume rm alpamayo-hf-cache alpamayo-torch-cache 2>/dev/null || true
	@echo "Removed cached model volumes"
