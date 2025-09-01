# Etude LLMv0.2

## 项目简介

Etude LLM是一个轻量级的语言模型实现项目，旨在提供一个可定制、可扩展的语言模型架构。该项目采用模块化设计，支持多种模型变体，包括标准Transformer结构和混合专家模型(MoE)。项目名称"Etude"（练习曲）寓意该项目既是语言模型实现的学习实践，也可作为更复杂模型架构的基础。

## 项目结构

```
Etude LLM/
├── inference/          # 模型推理代码
│   ├── inference.py    # 基础推理实现
├── model/              # 模型定义
│   ├── model.py        # 基础模型架构 
├── tool/               # 数据处理工具
│   ├── cut_jsonl.py    # JSONL数据处理
│   ├── cut_jsonl_sft.py # SFT数据格式处理
│   ├── cut_txt.py      # 文本切分工具
│   └── extract_xml.py  # XML数据提取工具
├── train/              # 训练相关代码
│   ├── full_sft_train.py # 全量SFT训练
│   ├── tarin_semanteme.py# 语义训练
|   └──train_tokenizer.py# 训练分词器
├── training_data/      # 训练数据
│   ├── big_json/       # 大型JSON格式数据
│   ├── big_text/       # 大型文本数据
│   ├── text/           # 文本数据
│   └── xml/            # XML格式数据
└── weight/             # 模型权重
    ├── full_sft_weight/ # 全量SFT模型权重
    └── semanteme_weight/ # 语义模型权重
```

## 核心功能

### 模型架构

Etude LLM实现了以下核心组件：

1. **注意力机制**：
   - 多头注意力(MultiHeadAttention)

2. **Transformer块**：
   - 标准前馈网络(FeedForward)
   - 支持混合专家模型(MoE)的块结构

3. **混合专家(MoE)实现**：
   - 路由器(MOERouter)：决定每个token应该由哪些专家处理
   - 稀疏MoE(SparseMOE)：支持每个token只通过部分专家进行处理


### 数据处理

项目提供多种数据处理工具：

- **XML处理**：从维基百科等XML格式数据中提取和清洗文本
- **文本切分**：将大型文本文件切分为训练样本
- **JSONL处理**：处理JSON格式的训练数据

### 训练方法

支持多种训练范式：

- **全量微调(Full Fine-tuning)**：适用于有足够计算资源的场景
- **语义训练**：针对语义理解的专门训练方法



## 技术特点

1. **模块化设计**：各组件高度解耦，便于扩展和实验
2. **灵活配置**：通过`EtudeConfig`类提供统一的配置接口
3. **混合专家机制**：实现了高效的条件计算路径


## 使用场景

- 语言模型研究与实验
- 文本生成应用
- 自然语言处理任务
- 作为更复杂模型开发的起点

## 未来发展

- 支持更多模型架构变体
- 扩展到多模态任务
- 优化推理速度
- 加入webUI
- 重构代码

## 总结

- 相比之前的复刻gpt2，这个要好很多，但仍然极其稀烂，未来我将会重构代码。
- 虽然这个版本已经可以跑起来，但是很垃圾，训练速度极其慢。
- moe我没过多测试，放在那就是意思意思的。

