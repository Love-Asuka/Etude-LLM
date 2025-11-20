# Etude LLMv0.2

![visitors](https://visitor-badge.laobi.icu/badge?page_id=Etude-AI/Etude-LLM)
[![GitHub Repo stars](https://img.shields.io/github/stars/Etude-AI/Etude-LLM?style=social)](https://github.com/Etude-AI/Etude-LLM/stargazers)
[![GitHub Code License](https://img.shields.io/github/license/Etude-AI/Etude-LLM)](LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/Etude-AI/Etude-LLM)](https://github.com/Etude-AI/Etude-LLM/commits/main)
[![GitHub pull request](https://img.shields.io/badge/PRs-welcome-blue)](https://github.com/Etude-AI/Etude-LLM/pulls)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/Etude-AI/Etude-LLMv0.2)

> **注意**: 本文档为机器翻译版本。如需查看原始中文文档，请访问 [README_CN.md](README_CN.md)

## Project Overview

Etude LLM is a lightweight language model implementation project designed to provide a customizable and extensible language model architecture. The project adopts a modular design and supports standard Transformer structures. The project name "Etude" (Etude) symbolizes that this project is both a learning practice for language model implementation and can serve as a foundation for more complex model architectures.



![Etude](./img/Etude.gif)

## Core Features

### Model Architecture

Etude LLM implements the following core components:

1. **Attention Mechanism**:
   - Multi-Head Attention (MultiHeadAttention)

2. **Transformer Blocks**:
   - Standard Feed-Forward Network (FeedForward)

### Training Methods

Supports multiple training paradigms:

- **Pretraining**: Basic language model training
- **Supervised Fine-Tuning (SFT)**: Instruction fine-tuning training
- **Tokenizer Training**: Custom tokenizer training

## Quick Start

### Training Process Overview

The Etude LLM training process consists of three main steps:

1. **Tokenizer Training** - Create custom tokenizer
2. **Pretraining** - Basic language model training
3. **Supervised Fine-Tuning (SFT)** - Instruction fine-tuning training

### Detailed Training Steps

#### 1. Tokenizer Training

First, train a custom tokenizer:

```bash
cd train
python train_tokenizer.py all
```

This will:
- Validate configuration files
- Train BPE tokenizer
- Create Hugging Face compatible tokenizer configuration
- Validate tokenizer functionality

#### 2. Pretraining

Perform basic language model pretraining:

```bash
cd train
python train_pretrain.py
```

Configuration parameters (can be adjusted in `config.py`):
- Batch size: 16
- Learning rate: 3e-4
- Training epochs: 3
- Device: Auto-detect CUDA/CPU

#### 3. Supervised Fine-Tuning (SFT)

Perform instruction fine-tuning on the pretrained model:

```bash
cd train
python train_sft.py
```

Configuration parameters:
- Batch size: 8
- Learning rate: 3e-5
- Training epochs: 3

### Model Saving

After training completion, models will be saved in:
- Pretrained model: `weight/etude_pretrained_model/`
- SFT model: `weight/etude_sft_model/`
- Tokenizer: `weight/tokenizer/`

### Resume Training

All training scripts support checkpoint resumption:
- Automatically detect checkpoint files
- Restore optimizer state and training progress
- Support continuing SFT training from pretrained models

### Model Inference

After training completion, you can use inference scripts for model inference:

```bash
cd inference
python inference.py
```

This script will load the model directly from the saved model configuration files without relying on training configuration classes.

Additionally, the project provides Hugging Face compatible inference scripts:

```bash
cd inference
python hf.py
```

This script uses the Hugging Face Transformers library to load the model and supports streaming output and chat templates.

### Hugging Face Format Conversion

The project provides a tool script to convert trained Etude models to Hugging Face compatible format:

```bash
cd tool
python convert_hf.py
```

This script will convert Etude model weights and configurations to Llama format, making the model compatible with the Hugging Face ecosystem.

## Technical Features

1. **Modular Design**: Highly decoupled components for easy extension and experimentation
2. **Flexible Configuration**: Unified configuration interface through `EtudeConfig` class

## Use Cases

- Language model research and experimentation
- Text generation applications
- Natural language processing tasks
- Starting point for more complex model development

## Future Development

- Support for more model architecture variants
- Expansion to multimodal tasks
- Inference speed optimization
- Code refactoring

## Summary

- Compared to the previous GPT-2 replication, this version is much better, but still needs significant improvement. Future versions will include code refactoring.
- Although this version can run, the training speed is extremely slow.
- The original v0.1 version could not engage in normal conversation, so it was not preserved.

---

**原始中文文档请参考：[README_CN.md](README_CN.md)**