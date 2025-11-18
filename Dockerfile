# =============
# BUILDER STAGE
# =============
FROM python:3.13-slim-bookworm AS builder

# Install curl and certificates for uv installer
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv
## Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh
## Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh
## Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Set working directory
WORKDIR /app

# directory for uv virtual environment 
ENV PATH="/app/.venv/bin:$PATH"
# Copy dependency files 
COPY pyproject.toml uv.lock .python-version ./

# Install dependencies
RUN uv sync --locked

# =============
# FINAL STAGE
# =============
FROM python:3.13-slim-bookworm AS final

## Open Container Initiative (OCI) Standard Metadata
LABEL org.opencontainers.image.title="steam_game_predictor" \
    org.opencontainers.image.version="1.0.0" \
    org.opencontainers.image.description="Image for predictor of game success based on publicly available Steam game metadata" \
    org.opencontainers.image.authors="Sergio Mosquim Junior <sergiomosquim@live.com>" \
    org.opencontainers.image.source="https://github.com/sergiomosquim/steam-game-predictor" \
    org.opencontainers.image.licenses="MIT"

# Define user details
ARG USER_NAME=steam_predictor
ARG USER_UID=1000
ARG USER_GID=1000

## Set up User, Work Directory, and Environment
RUN groupadd -g ${USER_GID} ${USER_NAME} && \
    useradd -m -u ${USER_UID} -g ${USER_GID} --create-home ${USER_NAME}

# Copy the environment from builder
COPY --from=builder /app/.venv /app/.venv
# directory for uv virtual environment 
ENV PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Now copy the model and required code
COPY predict_docker.py transformers.py final_xgb_model.pkl log_score_quantiles.json runapp.sh ./

# Set up entrypoint
RUN chmod +x runapp.sh

USER ${USER_NAME}
# expose port 9696 set in the FastAPI
EXPOSE 9696
ENTRYPOINT ["./runapp.sh"]