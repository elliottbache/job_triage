FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        latexmk \
        texlive-fonts-extra \
        texlive-latex-extra \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir .

ENTRYPOINT ["job_triage"]
