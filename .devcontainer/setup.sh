#!/bin/bash

git submodule update --init --recursive
git submodule update --recursive --remote

pip install --upgrade pip
pip install -e ./nsdc_framework
pip install -e ./wfcommons