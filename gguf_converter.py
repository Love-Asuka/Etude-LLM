#!/usr/bin/env python3
"""
GGUF Converter for Etude Architecture
将Etude自研架构模型转换为GGUF格式，支持llama.cpp推理
"""

import os
import sys
import json
import struct
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import torch
from transformers import AutoTokenizer, AutoConfig

# 尝试导入gguf库
try:
    import gguf
except ImportError:
    print("错误: 需要安装gguf库")
    print("请运行: pip install gguf")
    sys.exit(1)

# 导入本地模型
try:
    from model.model import Etude, EtudeHFConfig
except ImportError:
    print("错误: 无法导入Etude模型")
    print("请确保model/model.py文件存在且可访问")
    sys.exit(1)


class EtudeGGUFConverter:
    """Etude架构到GGUF格式的转换器"""
    
    def __init__(self, model_path: str, output_path: str, arch: str = "etude"):
        """
        初始化转换器
        
        Args:
            model_path: 输入模型路径（PyTorch模型）
            output_path: 输出GGUF文件路径
            arch: 架构名称，默认为"etude"
        """
        self.model_path = Path(model_path)
        self.output_path = Path(output_path)
        self.arch = arch
        
        # 验证输入路径
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型路径不存在: {model_path}")
        
        # 创建输出目录
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 初始化变量
        self.model = None
        self.config = None
        self.tokenizer = None
        self.gguf_writer = None
        
    def load_model(self) -> None:
        """加载PyTorch模型和配置"""
        print(f"正在加载模型: {self.model_path}")
        
        try:
            # 加载配置
            config_path = self.model_path / "config.json"
            if config_path.exists():
                self.config = EtudeHFConfig.from_pretrained(self.model_path)
            else:
                print("警告: 未找到config.json，使用默认配置")
                self.config = EtudeHFConfig()
            
            # 加载模型
            self.model = Etude.from_pretrained(self.model_path)
            self.model.eval()
            
            # 加载分词器
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            except Exception as e:
                print(f"警告: 无法加载分词器: {e}")
                self.tokenizer = None
                
            print(f"模型加载成功:")
            print(f"  - 词汇表大小: {self.config.vocab_size}")
            print(f"  - 层数: {self.config.n_layer}")
            print(f"  - 注意力头数: {self.config.n_head}")
            print(f"  - 嵌入维度: {self.config.n_embd}")
            
        except Exception as e:
            raise RuntimeError(f"加载模型失败: {e}")
    
    def _get_tensor_name_mapping(self) -> Dict[str, str]:
        """
        获取PyTorch张量名称到GGUF张量名称的映射
        
        Returns:
            张量名称映射字典
        """
        mapping = {}
        
        # Token embedding
        mapping["token_embedding.weight"] = "token_embd.weight"
        
        # 输出层
        mapping["lm_head.weight"] = "output.weight"
        
        # 最终层归一化
        mapping["ln_f.weight"] = "output_norm.weight"
        
        # Transformer blocks
        for i in range(self.config.n_layer):
            # 注意力层归一化
            mapping[f"blocks.{i}.ln1.weight"] = f"blk.{i}.attn_norm.weight"
            
            # 注意力投影层
            mapping[f"blocks.{i}.att.qkv_proj.weight"] = f"blk.{i}.attn_qkv.weight"
            mapping[f"blocks.{i}.att.out_proj.weight"] = f"blk.{i}.attn_output.weight"
            
            # FFN层归一化
            mapping[f"blocks.{i}.ln2.weight"] = f"blk.{i}.ffn_norm.weight"
            
            # FFN层
            mapping[f"blocks.{i}.ffn.net.w1.weight"] = f"blk.{i}.ffn_gate.weight"
            mapping[f"blocks.{i}.ffn.net.w2.weight"] = f"blk.{i}.ffn_down.weight"
            mapping[f"blocks.{i}.ffn.net.w3.weight"] = f"blk.{i}.ffn_up.weight"
        
        return mapping
    
    def _convert_tensor_dtype(self, tensor: torch.Tensor) -> Tuple[np.ndarray, int]:
        """
        转换张量数据类型为GGUF支持的格式
        
        Args:
            tensor: PyTorch张量
            
        Returns:
            (numpy数组, GGUF数据类型)
        """
        # 转换为numpy
        if tensor.dtype == torch.float32:
            return tensor.detach().cpu().numpy().astype(np.float32), gguf.GGMLQuantizationType.F32
        elif tensor.dtype == torch.float16:
            return tensor.detach().cpu().numpy().astype(np.float16), gguf.GGMLQuantizationType.F16
        elif tensor.dtype == torch.bfloat16:
            # GGUF不直接支持bfloat16，转换为float32
            return tensor.detach().cpu().float().numpy().astype(np.float32), gguf.GGMLQuantizationType.F32
        else:
            # 默认转换为float32
            return tensor.detach().cpu().float().numpy().astype(np.float32), gguf.GGMLQuantizationType.F32
    
    def _set_gguf_metadata(self) -> None:
        """设置GGUF文件的元数据"""
        print("设置GGUF元数据...")
        
        # 基础架构信息
        self.gguf_writer.add_architecture(self.arch)
        self.gguf_writer.add_name("Etude")
        self.gguf_writer.add_description("Etude自研架构语言模型")
        
        # 模型参数
        self.gguf_writer.add_context_length(4096)  # 默认上下文长度
        self.gguf_writer.add_embedding_length(self.config.n_embd)
        self.gguf_writer.add_block_count(self.config.n_layer)
        self.gguf_writer.add_feed_forward_length(int(self.config.n_embd * 4 * (2 / 3)))
        self.gguf_writer.add_head_count(self.config.n_head)
        self.gguf_writer.add_head_count_kv(self.config.n_head)  # Etude使用相同的KV头数
        
        # RoPE参数
        self.gguf_writer.add_rope_dimension_count(self.config.n_embd // self.config.n_head)
        self.gguf_writer.add_rope_freq_base(10000.0)
        
        # 归一化参数
        self.gguf_writer.add_layer_norm_rms_eps(1e-6)
        
        # 词汇表信息
        if self.tokenizer is not None:
            self._add_tokenizer_metadata()
        else:
            # 如果没有分词器，添加基本词汇表信息
            self.gguf_writer.add_vocab_size(self.config.vocab_size)
            
        # 特殊token
        self.gguf_writer.add_bos_token_id(self.config.eos_token_id)  # 使用eos作为bos
        self.gguf_writer.add_eos_token_id(self.config.eos_token_id)
        self.gguf_writer.add_pad_token_id(self.config.pad_token_id)
        
        # 文件类型
        self.gguf_writer.add_file_type(gguf.GGMLQuantizationType.F32)
        
        print("元数据设置完成")
    
    def _add_tokenizer_metadata(self) -> None:
        """添加分词器相关的元数据"""
        if self.tokenizer is None:
            return
            
        print("添加分词器元数据...")
        
        # 词汇表大小
        vocab_size = len(self.tokenizer.get_vocab())
        self.gguf_writer.add_vocab_size(vocab_size)
        
        # 获取词汇表
        vocab = self.tokenizer.get_vocab()
        
        # 按索引排序词汇表
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
        
        # 添加tokens
        tokens = []
        scores = []
        token_types = []
        
        for token, idx in sorted_vocab:
            tokens.append(token.encode('utf-8'))
            scores.append(0.0)  # 默认分数
            token_types.append(gguf.TokenType.NORMAL)
        
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(token_types)
        
        # 分词器类型
        self.gguf_writer.add_tokenizer_model("llama")  # 使用llama兼容的分词器类型
        
        print(f"分词器元数据添加完成，词汇表大小: {vocab_size}")
    
    def _initialize_gguf_writer(self) -> None:
        """初始化GGUF写入器"""
        print(f"初始化GGUF写入器: {self.output_path}")
        self.gguf_writer = gguf.GGUFWriter(str(self.output_path), self.arch)
    
    def _add_special_tokens(self) -> None:
        """添加特殊token的定义"""
        # 这里可以根据需要添加特殊token的处理
        # 例如：<|endoftext|>, <|pad|> 等
        pass
    
    def _convert_weights(self) -> None:
        """转换模型权重到GGUF格式"""
        print("开始转换模型权重...")
        
        # 获取张量名称映射
        tensor_mapping = self._get_tensor_name_mapping()
        
        # 获取模型状态字典
        state_dict = self.model.state_dict()
        
        converted_count = 0
        total_count = len(state_dict)
        
        for pytorch_name, tensor in state_dict.items():
            # 获取对应的GGUF张量名称
            gguf_name = tensor_mapping.get(pytorch_name)
            
            if gguf_name is None:
                print(f"警告: 未找到张量映射 {pytorch_name}")
                continue
            
            # 处理特殊的张量转换
            converted_tensor = self._process_tensor(pytorch_name, tensor)
            
            # 转换数据类型
            numpy_tensor, ggml_type = self._convert_tensor_dtype(converted_tensor)
            
            # 添加到GGUF文件
            self.gguf_writer.add_tensor(gguf_name, numpy_tensor, ggml_type)
            
            converted_count += 1
            if converted_count % 10 == 0:
                print(f"已转换 {converted_count}/{total_count} 个张量")
        
        print(f"权重转换完成，共转换 {converted_count} 个张量")
    
    def _process_tensor(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        """
        处理特定张量的转换逻辑
        
        Args:
            name: 张量名称
            tensor: 原始张量
            
        Returns:
            处理后的张量
        """
        # QKV投影层需要特殊处理
        if "qkv_proj.weight" in name:
            return self._process_qkv_weight(tensor)
        
        # 其他张量直接返回
        return tensor
    
    def _process_qkv_weight(self, qkv_weight: torch.Tensor) -> torch.Tensor:
        """
        处理QKV投影权重
        
        Etude模型使用单个线性层生成Q、K、V，需要分离为独立的权重
        
        Args:
            qkv_weight: QKV投影权重 [3*n_embd, n_embd]
            
        Returns:
            重新排列的权重张量
        """
        n_embd = self.config.n_embd
        n_head = self.config.n_head
        head_dim = n_embd // n_head
        
        # 分离Q、K、V权重
        q_weight = qkv_weight[:n_embd, :]  # [n_embd, n_embd]
        k_weight = qkv_weight[n_embd:2*n_embd, :]  # [n_embd, n_embd]
        v_weight = qkv_weight[2*n_embd:, :]  # [n_embd, n_embd]
        
        # 重新组织为 [n_head, 3, head_dim, n_embd] 然后展平
        # 这样可以确保Q、K、V在内存中是交错排列的，符合llama.cpp的期望
        qkv_interleaved = torch.zeros_like(qkv_weight)
        
        for head in range(n_head):
            start_idx = head * head_dim
            end_idx = (head + 1) * head_dim
            
            # 对于每个头，按Q、K、V的顺序排列
            qkv_interleaved[head * 3 * head_dim:(head + 1) * 3 * head_dim, :] = torch.cat([
                q_weight[start_idx:end_idx, :],
                k_weight[start_idx:end_idx, :],
                v_weight[start_idx:end_idx, :]
            ], dim=0)
        
        return qkv_interleaved
    
    def convert(self) -> None:
        """执行完整的转换流程"""
        try:
            print("=" * 50)
            print("开始Etude模型到GGUF格式转换")
            print("=" * 50)
            
            # 1. 加载模型
            self.load_model()
            
            # 2. 初始化GGUF写入器
            self._initialize_gguf_writer()
            
            # 3. 设置元数据
            self._set_gguf_metadata()
            
            # 4. 转换权重
            self._convert_weights()
            
            # 5. 写入文件
            print("正在写入GGUF文件...")
            self.gguf_writer.write_header_to_file()
            self.gguf_writer.write_kv_data_to_file()
            self.gguf_writer.write_tensors_to_file()
            
            print("=" * 50)
            print(f"转换完成！输出文件: {self.output_path}")
            print(f"文件大小: {self.output_path.stat().st_size / (1024*1024):.2f} MB")
            print("=" * 50)
            
        except Exception as e:
            print(f"转换失败: {e}")
            raise
        finally:
            # 清理资源
            if self.gguf_writer:
                self.gguf_writer.close()
    
    def validate_conversion(self) -> bool:
        """验证转换结果"""
        try:
            print("验证GGUF文件...")
            
            # 检查文件是否存在
            if not self.output_path.exists():
                print("错误: 输出文件不存在")
                return False
            
            # 检查文件大小
            file_size = self.output_path.stat().st_size
            if file_size == 0:
                print("错误: 输出文件为空")
                return False
            
            # 尝试读取GGUF文件头
            try:
                with open(self.output_path, 'rb') as f:
                    # 读取GGUF魔数
                    magic = f.read(4)
                    if magic != b'GGUF':
                        print("错误: 不是有效的GGUF文件")
                        return False
            except Exception as e:
                print(f"错误: 无法读取文件头: {e}")
                return False
            
            print("GGUF文件验证通过")
            return True
            
        except Exception as e:
             print(f"验证过程出错: {e}")
             return False


def create_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="将Etude自研架构模型转换为GGUF格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本转换
  python gguf_converter.py --input ./etude_model --output ./etude_model.gguf
  
  # 指定架构名称
  python gguf_converter.py --input ./etude_model --output ./etude_model.gguf --arch etude-v1
  
  # 转换后验证
  python gguf_converter.py --input ./etude_model --output ./etude_model.gguf --validate
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="输入模型路径（包含PyTorch模型文件的目录）"
    )
    
    parser.add_argument(
        "--output", "-o", 
        type=str,
        required=True,
        help="输出GGUF文件路径"
    )
    
    parser.add_argument(
        "--arch",
        type=str,
        default="etude",
        help="架构名称（默认: etude）"
    )
    
    parser.add_argument(
        "--validate",
        action="store_true",
        help="转换后验证GGUF文件"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="显示详细输出"
    )
    
    return parser


def validate_args(args) -> None:
    """验证命令行参数"""
    # 检查输入路径
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"输入路径不存在: {args.input}")
    
    if not input_path.is_dir():
        raise ValueError(f"输入路径必须是目录: {args.input}")
    
    # 检查是否包含必要的模型文件
    required_files = ["pytorch_model.bin", "model.safetensors", "config.json"]
    has_model_file = any((input_path / f).exists() for f in required_files[:2])
    
    if not has_model_file:
        print("警告: 未找到标准的模型文件 (pytorch_model.bin 或 model.safetensors)")
        print("将尝试直接加载模型...")
    
    # 检查输出路径
    output_path = Path(args.output)
    if output_path.exists():
        response = input(f"输出文件已存在: {args.output}\n是否覆盖? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("转换已取消")
            sys.exit(0)
    
    # 创建输出目录
    output_path.parent.mkdir(parents=True, exist_ok=True)


def main():
    """主函数"""
    try:
        # 解析命令行参数
        parser = create_parser()
        args = parser.parse_args()
        
        # 验证参数
        validate_args(args)
        
        # 设置日志级别
        if args.verbose:
            print("详细模式已启用")
        
        # 创建转换器
        converter = EtudeGGUFConverter(
            model_path=args.input,
            output_path=args.output,
            arch=args.arch
        )
        
        # 执行转换
        converter.convert()
        
        # 验证结果
        if args.validate:
            if converter.validate_conversion():
                print("✓ GGUF文件验证成功")
            else:
                print("✗ GGUF文件验证失败")
                sys.exit(1)
        
        print("\n转换完成！")
        print(f"输出文件: {args.output}")
        
        # 显示使用建议
        print("\n使用建议:")
        print("1. 可以使用llama.cpp加载此GGUF文件进行推理")
        print("2. 建议先用小批量数据测试模型输出的正确性")
        print("3. 如需量化，可使用llama.cpp的量化工具")
        
    except KeyboardInterrupt:
        print("\n转换被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n错误: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()