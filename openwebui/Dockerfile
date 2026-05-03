FROM ghcr.io/open-webui/open-webui:main

# Install poppler-utils for advanced PDF parsing
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*
