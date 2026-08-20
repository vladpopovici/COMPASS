FROM python:3.13.11-slim-trixie AS base

ARG dock_user=devil
ARG dock_group=devil
ARG dock_uid=1000
ARG dock_gid=1000

SHELL ["/usr/bin/bash", "-c"]
RUN chsh -s /usr/bin/bash root

RUN mkdir /.local; chmod ago+rwx /.local
RUN mkdir /.cache; chmod ago+rwx /.cache
RUN mkdir /.config; chmod ago+rwx /.config

ENV PATH="/.local/bin:${PATH}"
ENV HOSTNAME="devil"
ENV DEBIAN_FRONTEND=noninteractive

RUN set -eux; \
	apt-get update; \
	apt-get install -y --no-install-recommends \
    build-essential \
    autoconf automake gdb git libffi-dev zlib1g-dev libssl-dev \
    gfortran \
    libopenblas-pthread-dev liblapack64-dev \
    libhdf5-dev libzck-dev libzip-dev zlib1g-dev libblosc2-dev \
    comerr-dev \
		mc \
	; \
	apt-get dist-clean

RUN groupadd -g ${dock_gid} ${dock_group}
RUN useradd -ms /bin/bash -g ${dock_gid} -u ${dock_uid} ${dock_user}
RUN chsh -s /usr/bin/bash ${dock_user}

ENV PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100 \
  # Poetry's configuration:
  POETRY_NO_INTERACTION=1 \
  POETRY_VIRTUALENVS_CREATE=1 \
  POETRY_VIRTUALENVS_IN_PROJECT=1 \
  POETRY_CACHE_DIR='/home/devil/.cache/pypoetry' \
  POETRY_VERSION=2.2.1

USER ${dock_user}
WORKDIR /home/${dock_user}

ENV PATH="/home/${dock_user}/.local/bin:${PATH}"
RUN echo "export PATH=${PATH}" >> /home/${dock_user}/.bashrc

RUN pip install poetry==${POETRY_VERSION}
RUN pip install ipython
RUN pip install jupyterlab ipywidgets


# Prepare for VSCode
FROM base

SHELL ["/usr/bin/bash", "-c"]

USER root
RUN mkdir -p /.vscode-server; chmod ago+xrw /.vscode-server

# Port for jupyter-lab
EXPOSE 8888

USER ${dock_user}
ENV HOME=/home/${dock_user}
RUN mkdir -p ${HOME}/.vscode-server


