FROM gitpod/workspace-full
ENV PYTHONUSERBASE=/workspace/.pip-modules
ENV PATH=$PYTHONUSERBASE/bin:$PATH
ENV PIP_USER=yes

RUN sudo apt-get update && sudo apt-get install -y graphviz graphviz-dev
RUN pip install --upgrade pip
