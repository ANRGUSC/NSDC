# FROM jaredraycoleman/workspace-full-tk-vnc
FROM gitpod/workspace-full

RUN sudo apt-get update && sudo apt-get install -y graphviz graphviz-dev
