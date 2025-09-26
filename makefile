IMAGE_API=ghcr.io/$(shell echo ${GITHUB_REPOSITORY} | tr '[:upper:]' '[:lower:]')-api
IMAGE_WEB=ghcr.io/$(shell echo ${GITHUB_REPOSITORY} | tr '[:upper:]' '[:lower:]')-web
TAG?=latest

.PHONY: up down build push test

up:
\tdocker compose up --build

down:
\tdocker compose down

build:
\tdocker build -f Dockerfile.api -t $(IMAGE_API):$(TAG) .
\tdocker build -f streamlit/Dockerfile -t $(IMAGE_WEB):$(TAG) ./streamlit

push:
\tdocker push $(IMAGE_API):$(TAG)
\tdocker push $(IMAGE_WEB):$(TAG)

test:
\tpytest -q || true
