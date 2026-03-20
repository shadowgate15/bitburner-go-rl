# Build argument: base64-encoded bytes representing the Uint8Array save value.
# Pass with: --build-arg SAVE_DATA=$(base64 -w0 your_save_file)
ARG SAVE_DATA=""

FROM mcr.microsoft.com/playwright/python:v1.50.0-noble

WORKDIR /app

# Re-declare ARG after FROM so it is visible in this build stage.
ARG SAVE_DATA=""

# Bake the save data into the image at build time.
RUN printf '%s' "${SAVE_DATA}" > /app/save_data.b64

# Copy the environment setup script.
COPY docker/setup_game_env.py /app/setup_game_env.py

# On container start: load the Bitburner page, populate IndexDB, and keep
# the browser alive so the RL training process can connect.
CMD ["python", "/app/setup_game_env.py"]
