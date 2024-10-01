<div align="center">

# Distract Large Language Models for Automatic Jailbreak Attack

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
[![Paper](http://img.shields.io/badge/paper-arxiv.2403.08424-B31B1B.svg)](https://arxiv.org/abs/2403.08424)
[![Conference](http://img.shields.io/badge/EMNLP-2024-4b44ce.svg)](https://2024.emnlp.org/)

</div>

## Description
Official implementation of the paper "Distract Large Language Models for Automatic Jailbreak Attack" (EMNLP 2024).

A black-box and optimization-based method to automatically generate jailbreak templates for large language models, suggesting that the LLM's tendency to become distracted may lead to its susceptibility to jailbreak attempts.

## Installation

```bash
# clone project
git clone https://github.com/sufenlp/AttanttionShiftJailbreak
cd AttanttionShiftJailbreak

# create conda environment and activate it
conda create -n jailbreak python=3.10
conda activate jailbreak

# install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## How to run

Step 1: Set OpenAI api_key or huggingface models' path in `src/config.py`.

Step 2: Download our judge model from [here](https://huggingface.co/zgxiao/deberta-v3-large-judge-model).

Step 3 :Train with default configuration:

```bash
bash scripts/schedule.sh
```

All the hyperparameters can be found in `src/main.py`. You may need a [wandb](https://wandb.ai/site) account to log the training process.

## Acknowledgements
This project builds upon the work of the following open-source projects:

- [JailbreakingLLMs](https://github.com/patrickrchao/JailbreakingLLMs)
- [DeepInception](https://github.com/tmlr-group/DeepInception)

We are grateful for their contributions to the field and the availability of their code, which served as a foundation for our work.

## Citation

```bibtex
@article{xiao2024tastle,
  title={Tastle: Distract large language models for automatic jailbreak attack},
  author={Xiao, Zeguan and Yang, Yan and Chen, Guanhua and Chen, Yun},
  journal={arXiv preprint arXiv:2403.08424},
  year={2024}
}
```