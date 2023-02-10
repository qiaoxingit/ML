FROM continuumio/miniconda3:22.11.1-alpine as development
COPY mlenv.yml .
RUN conda env create -f mlenv.yml