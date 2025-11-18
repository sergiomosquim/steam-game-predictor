#!/usr/bin/env bash
set -e

uvicorn predict_docker:app --host 0.0.0.0 --port 9696