# edge-desk always-on live runner (backlog #66c). Read-only market data + PAPER trades only.
# The repo itself is NOT baked in: it is a git clone on the host (/srv/edge-desk) mounted at the
# same path, so run.sh can `git pull` new code and push data commits like the Actions live job.
# Build from the repo root: `cd /srv/edge-desk && docker build -t edge-desk:latest -f deploy/Dockerfile .`
FROM python:3.11-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends git openssh-client ca-certificates tzdata \
 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
ENV PYTHONUNBUFFERED=1 EDGE_RUNNER=vps REPO=/srv/edge-desk
COPY deploy/run.sh /usr/local/bin/edge-desk-run
RUN chmod +x /usr/local/bin/edge-desk-run
CMD ["/usr/local/bin/edge-desk-run"]
