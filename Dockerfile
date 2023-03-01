FROM continuumio/miniconda3:22.11.1-alpine as development
RUN apk update && apk add --no-cache bash git
