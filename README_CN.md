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

## 项目简介

Etude LLM是一个轻量级的语言模型实现项目，旨在提供一个可定制、可扩展的语言模型架构。该项目采用模块化设计，支持标准Transformer结构。项目名称"Etude"（练习曲）寓意该项目既是语言模型实现的学习实践，也可作为更复杂模型架构的基础。



![Etude](./img/Etude.gif)

## 核心功能

### 模型架构

Etude LLM实现了以下核心组件：

1. **注意力机制**：
   - 多头注意力(MultiHeadAttention)

2. **Transformer块**：
   - 标准前馈网络(FeedForward)

### 数据处理

项目提供多种数据处理工具：

- **XML处理**：从维基百科等XML格式数据中提取和清洗文本
- **文本切分**：将大型文本文件切分为训练样本
- **JSONL处理**：处理JSON格式的训练数据

### 训练方法

支持多种训练范式：

- **预训练(Pretraining)**：基础语言模型训练
- **监督微调(SFT)**：指令微调训练
- **分词器训练**：自定义分词器训练

## 快速开始

### 训练流程概述

Etude LLM的训练流程分为三个主要步骤：

1. **分词器训练** - 创建自定义分词器
2. **预训练** - 基础语言模型训练
3. **监督微调(SFT)** - 指令微调训练

### 详细训练步骤

#### 1. 分词器训练

首先训练自定义分词器：

```bash
cd train
python train_tokenizer.py all
```

这将：
- 验证配置文件
- 训练BPE分词器
- 创建Hugging Face兼容的分词器配置
- 验证分词器功能

#### 2. 预训练

进行基础语言模型预训练：

```bash
cd train
python train_pretrain.py
```

配置参数（可在`config.py`中调整）：
- 批量大小：16
- 学习率：3e-4
- 训练轮数：3
- 设备：自动检测CUDA/CPU

#### 3. 监督微调(SFT)

在预训练模型基础上进行指令微调：

```bash
cd train
python train_sft.py
```

配置参数：
- 批量大小：8
- 学习率：3e-5
- 训练轮数：3

### 数据准备
训练数据应放置在以下目录：

示例：

- 预训练数据：`training_data/pretrain/pretrain_hq.jsonl`
- SFT数据：`training_data/sft/sft_mini_512.jsonl`

### 模型保存

训练完成后，模型将保存在：
- 预训练模型：`weight/etude_pretrained_model/`
- SFT模型：`weight/etude_sft_model/`
- 分词器：`weight/tokenizer/`

### 恢复训练

所有训练脚本支持断点续训：
- 自动检测检查点文件
- 恢复优化器状态和训练进度
- 支持从预训练模型继续SFT训练

### 模型推理

训练完成后，可以使用推理脚本进行模型推理：

```bash
cd inference
python inference.py
```

该脚本会直接从保存的模型配置文件中加载模型，无需依赖训练时的配置类。

此外，项目还提供了Hugging Face兼容的推理脚本：

```bash
cd inference
python hf.py
```

该脚本使用Hugging Face Transformers库加载模型，支持流式输出和聊天模板。

### Hugging Face格式转换

项目提供了一个工具脚本，可以将训练好的Etude模型转换为Hugging Face兼容的格式：

```bash
cd tool
python convert_hf.py
```

该脚本会将Etude模型权重和配置转换为Llama格式，使得模型可以与Hugging Face生态系统兼容。

## 技术特点

1. **模块化设计**：各组件高度解耦，便于扩展和实验
2. **灵活配置**：通过`EtudeConfig`类提供统一的配置接口

## 使用场景

- 语言模型研究与实验
- 文本生成应用
- 自然语言处理任务
- 作为更复杂模型开发的起点

## 未来发展

- 支持更多模型架构变体
- 扩展到多模态任务
- 优化推理速度
- 重构代码

## 总结

- 相比之前的复刻gpt2，这个要好很多，但仍然极其稀烂，未来我将会重构代码。
- 虽然这个版本已经可以跑起来，但是很垃圾，训练速度极其慢。
- 原来那个v0.1压根没法正常对话，所以我没保留

---

**英文文档（机器翻译）请参考：[README.md](README.md)**