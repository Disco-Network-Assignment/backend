#!/bin/bash
# Deploy the backend to the EC2 box: copy the code, then build and (re)start the stack.
#
#   deploy/deploy.sh <host> [ssh-key]
#   e.g. deploy/deploy.sh 3.109.228.123 ~/.ssh/disco-backend.pem
#
# What gets copied: the repo root minus the virtualenv, caches, git metadata and local results.
# The repo-root .env goes along too: it carries OPENAI_API_KEY and FRONTEND_ORIGIN.
# deploy/.env on the server must define BACKEND_HOST (the sslip.io hostname Caddy serves).
set -euo pipefail

HOST="${1:?usage: deploy/deploy.sh <host> [ssh-key]}"
KEY="${2:-$HOME/.ssh/disco-backend.pem}"
REMOTE="ubuntu@$HOST"
REMOTE_DIR="/opt/disco/backend"
SSH="ssh -i $KEY -o StrictHostKeyChecking=accept-new $REMOTE"

cd "$(dirname "$0")/.."

echo "==> copying code to $REMOTE:$REMOTE_DIR"
tar --exclude=.venv --exclude=.git --exclude='__pycache__' --exclude=.ruff_cache \
    --exclude=.pytest_cache --exclude='evals/results' -czf - . \
  | $SSH "mkdir -p $REMOTE_DIR && tar -xzf - -C $REMOTE_DIR"

echo "==> building and starting the stack"
$SSH "cd $REMOTE_DIR && docker compose -f deploy/docker-compose.yml --env-file deploy/.env up -d --build --remove-orphans"

echo "==> waiting for the API (its port is only reachable inside the compose network)"
COMPOSE="docker compose -f deploy/docker-compose.yml --env-file deploy/.env"
PROBE="python -c \"import urllib.request; print(urllib.request.urlopen('http://localhost:8000/health').read().decode())\""
for _ in $(seq 1 30); do
  if $SSH "cd $REMOTE_DIR && $COMPOSE exec -T api $PROBE" 2>/dev/null; then
    echo "==> deployed: https://$(grep BACKEND_HOST deploy/.env | cut -d= -f2)/health"
    exit 0
  fi
  sleep 3
done
echo "API did not come up; check: $SSH 'cd $REMOTE_DIR && $COMPOSE logs api'"
exit 1
