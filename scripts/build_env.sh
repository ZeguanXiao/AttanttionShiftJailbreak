#!/bin/sh


conda create -n jailbreak python=3.10
conda activate jailbreak

# mac
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 -c pytorch

# linux
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
