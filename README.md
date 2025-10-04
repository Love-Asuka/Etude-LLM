# Etude LLMv0.2

<p align="center">
  <a href="https://huggingface.co/Etude-AI/Etude-LLMv0.2">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue" alt="Hugging Face">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  </a>
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python">
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-2.0%2B-red" alt="PyTorch">
  </a>
</p>

> **注意**: 本文档为机器翻译版本。如需查看原始中文文档，请访问 [README_CN.md](README_CN.md)

## Project Overview

Etude LLM is a lightweight language model implementation project designed to provide a customizable and extensible language model architecture. The project adopts a modular design and supports standard Transformer structures. The project name "Etude" (Etude) symbolizes that this project is both a learning practice for language model implementation and can serve as a foundation for more complex model architectures.

**Hugging Face Model Page**: https://huggingface.co/Etude-AI/Etude-LLMv0.2

![Etude](./img/Etude.gif)

## Project Structure

```
Etude LLM/
├── inference/              # Model inference code
│   ├── __init__.py
│   ├── hf.py               # Hugging Face compatible inference implementation
│   ├── inference.py        # Basic inference implementation
│   └── inference_laRA.py   # LoRA inference implementation
├── model/                  # Model definitions
│   ├── __init__.py
│   ├── model.py            # Basic model architecture
│   └── model_loRA.py       # LoRA model architecture
├── tool/                   # Data processing tools
│   ├── convert_hf.py       # Etude to Hugging Face format conversion tool
│   ├── cut_json.py         # JSON data processing
│   ├── cut_jsonl.py        # JSONL data processing
│   ├── cut_jsonl_sft.py    # SFT data format processing
│   ├── cut_txt.py          # Text splitting tool
│   └── extract_xml.py      # XML data extraction tool
├── train/                  # Training related code
│   ├── __init__.py
│   ├── config.py           # Training configuration
│   ├── data_utils.py       # Data utility functions
│   ├── train_pretrain.py   # Pretraining implementation
│   ├── train_sft.py        # SFT training implementation
│   └── train_tokenizer.py  # Tokenizer training
├── img/                    # Image resources
│   └── Etude.gif
├── training_data/          # Training data (excluded by .gitignore)
└── weight/                 # Model weights (excluded by .gitignore)
```

## Core Features

### Model Architecture

Etude LLM implements the following core components:

1. **Attention Mechanism**:
   - Multi-Head Attention (MultiHeadAttention)

2. **Transformer Blocks**:
   - Standard Feed-Forward Network (FeedForward)

### Data Processing

The project provides various data processing tools:

- **XML Processing**: Extract and clean text from XML format data such as Wikipedia
- **Text Splitting**: Split large text files into training samples
- **JSONL Processing**: Process JSON format training data

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

### Data Preparation

Training data should be placed in the following directories:

Examples:

- Pretraining data: `training_data/pretrain/pretrain_hq.jsonl`
- SFT data: `training_data/sft/sft_mini_512.jsonl`

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

Or use the Lora version inference script:

```bash
cd inference
python inference_laRA.py
```

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


