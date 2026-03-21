FROM mcr.microsoft.com/playwright/python:v1.50.0-noble

WORKDIR /app

# Copy the save file that will be imported into the game at startup.
COPY bitburnerSave.json.gz /app/bitburnerSave.json.gz

# Copy the environment setup script.
COPY docker/setup_game_env.py /app/setup_game_env.py

RUN pip install --no-cache-dir playwright && \
  playwright install --with-deps

# On container start: load the Bitburner page, import the save via the
# in-game Options → Import Game UI flow, and keep the browser alive so
# the RL training process can connect.
CMD ["python", "/app/setup_game_env.py"]
