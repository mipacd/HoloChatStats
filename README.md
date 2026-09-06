# HoloChatStats

HoloChatStats collects YouTube livestream chat and stream metadata, turns it
into monthly analytics, and presents the results through an interactive web
application and the Eri conversational assistant.

This repository is the AWS-native rewrite of the project. It runs against
Floci for locally emulated AWS services and is designed to deploy automatically
from GitHub Actions onto a persistent, self-hosted runner.

## Features

### Audience and membership analytics

- Common chatters and members across channels
- User-similarity heatmaps and interactive community graphs
- Active-user and membership gains and losses
- Chat leaderboards, exclusive viewers, engagement rates, and individual user
  message frequency
- Membership counts, membership percentages, and membership-rank summaries
- Channel recommendations based on audience overlap

### Stream and chat analytics

- Total, average, maximum, and monthly streaming hours
- Month-over-month streaming-hour changes
- Stream-frequency heatmaps and calendars with timezone support
- Chat language makeup and language-specific message rates
- Japanese-user percentages
- Content-similarity graphs, funny timestamps, highlights, and a live viewer
- Exportable charts and English, Japanese, and Korean interfaces

### Eri assistant

The optional Eri service uses an LLM to answer questions about HoloChatStats
data, call approved analytics tools, execute constrained queries, and generate
charts from returned data. It includes rate limiting, prompt filtering, query
limits, and a separately protected administrator mode.

### Managed ETL pipeline

- Scheduled channel discovery through the YouTube API
- Resumable chat downloads with S3 checkpoints
- SQS-backed scan, download, retry, and ingestion workers
- Dead-letter queues, worker heartbeats, and automatic recovery of stalled jobs
- Strict oldest-month-first backlog processing
- Atomic monthly publication from `user_data_current` to `user_data`
- An admin dashboard for channels, queue depth, failures, and ETL progress

A closed month is published only after all its jobs are terminal and every active channel has
completed a post-month discovery scan. The open month continues ingesting in
the background.

## Architecture

| Component | Implementation |
| --- | --- |
| Frontend | React, TypeScript, Vite, nginx, ECS |
| Analytics API | Flask/Gunicorn on an emulated EC2 instance |
| Eri assistant | FastAPI, LangGraph, OpenRouter-compatible API |
| ETL workers | Python AWS Lambda functions |
| Database | Floci RDS PostgreSQL 16 with pgvector |
| Cache | Floci ElastiCache/Redis-compatible service |
| Messaging | SQS queues and dead-letter queues |
| Raw checkpoints | S3 |
| Scheduling | EventBridge |
| Configuration | SSM Parameter Store and Secrets Manager |
| ETL administration | Lambda behind API Gateway, proxied at `/admin/` |

The public frontend is published on host port `80`, suitable for a cloudflared
origin. The `/admin/` route is restricted to loopback and private LAN addresses
and explicitly rejects Cloudflare tunnel headers.

## Repository layout

```text
common/       Shared ETL, AWS, database, parsing, and dispatch code
handlers/     Lambda entry points
infra/        Idempotent Floci/AWS provisioning and database migration tools
migrations/   Ordered PostgreSQL schema and data migrations
frontend/     React/Vite application
web/          Flask analytics API and legacy templates
llm_chat/     Eri FastAPI service
scripts/      Administration and Lambda invocation utilities
tests/        Repository and production-configuration checks
```

## Production deployment

Production uses a persistent self-hosted GitHub Actions runner because the
Floci data volume and the legacy PostgreSQL container live on that Docker host.

## Local development

Start Floci:

```bash
docker compose -f infra/docker-compose.yml up -d
```

Set development credentials in your shell—not in tracked files—and provision a
fresh stack without a legacy copy:

```bash
export DB_PASSWORD='development-only-password'
export YT_API_KEY='your-youtube-api-key'

python infra/deploy.py \
  --endpoint http://localhost:4566 \
  --lambda-endpoint http://floci:4566 \
  --create-rds \
  --db-internal-host floci \
  --channels-file web/channel.json \
  --backlog-floor 2026-07-01T00:00:00+00:00 \
  --skip-restore \
  --build-in-docker
```

For frontend-only development:

```bash
cd frontend
npm install
npm run dev
```

Vite serves the development UI using the API base configured in
`frontend/vite.config.ts`. A complete UI requires the Flask API and its database
dependencies to be reachable.

## Using HoloChatStats

- Open `http://<server>/` for the analytics application.
- Use the navigation menus to select an analysis, channel group, channel, month,
  language, or timezone.
- Open `/eri` to ask Eri questions or request a chart when the LLM service is
  configured.
- On the local network, open `http://<server-lan-ip>/admin/` to inspect ETL
  progress and manage channels. 

Useful operator commands:

```bash
# Check Lambda-to-database/AWS connectivity
python scripts/invoke.py migrate --action ping

# Verify the deployed schema
python scripts/invoke.py migrate --action verify_schema

# Inspect discovery behavior
python scripts/invoke.py discover --arg force=true

# Ask the month dispatcher to release eligible work
python scripts/invoke.py migrate --action dispatch

# Rediscover the emulator-specific underlying admin API URL
python scripts/admin_url.py --refresh
```

## Testing

```bash
python -m compileall -q common handlers infra llm_chat web tests
python -m unittest discover -s tests -v
docker compose -f infra/docker-compose.yml config --quiet
```

These checks also run before every GitHub Actions deployment.

## Data-source notice

HoloChatStats analyzes public livestream metadata and available chat replays.
Availability and completeness depend on YouTube API access, replay retention,
channel configuration, and upstream rate limits. Monthly data should be treated
as complete only after the ETL admin page reports that publication has finished.
