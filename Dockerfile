FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install . && useradd --system --uid 10001 analyst && mkdir -p /data && chown analyst /data

USER 10001
VOLUME /data
EXPOSE 8700
HEALTHCHECK --interval=15s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8700/healthz', timeout=2)" || exit 1

ENTRYPOINT ["filings-analyst", "--index", "/data/index"]
CMD ["serve", "--host", "0.0.0.0"]
