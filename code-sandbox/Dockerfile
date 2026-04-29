FROM dhi.io/python:3.13-debian13-sfw-dev AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir \
    numpy \
    sympy \
    scipy \
    matplotlib \
    pandas \
    -r requirements.txt

FROM dhi.io/python:3.13

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY server.py .

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
