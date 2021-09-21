FROM gitpod/workspace-full

USER root
RUN apt-get update && apt-get install -y graphviz graphviz-dev

USER gitpod
RUN pyenv install 3.9.6 && pyenv global 3.9.6