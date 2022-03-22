FROM gitpod/workspace-full
RUN sudo apt-get update && sudo apt-get install -y graphviz graphviz-dev
RUN pip install --upgrade pip && \
    pip install -e ./nsdc_framework
