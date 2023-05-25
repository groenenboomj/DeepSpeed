# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import math
import torch
import torch.nn as nn
from ... import op_builder
from deepspeed import comm as dist
from deepspeed.utils.types import GATED_ACTIVATION_TYPES, ActivationFuncType
from deepspeed.accelerator import get_accelerator
from .op_binding import MLPGemmOp, VectorMatMulOp, GELUGemmOp, ResidualAddOp
from torch.autograd import Function

inference_cuda_module = None


class DeepSpeedMLPFunction(Function):
    @staticmethod
    def forward(ctx,
                input,
                residual,
                residual_norm,
                bias,
                inter_w,
                inter_b,
                attn_nw,
                attn_nb,
                config,
                mp_group,
                output_b,
                output_w,
                q_scales,
                q_groups,
                merge_count,
                mlp_gemm_func,
                fused_gemm_gelu,
                vector_matmul_func,
                bias_residual_func,
                residual_add_func,
                activation_func_type=ActivationFuncType.GELU):

        if attn_nw is None:
            output = fused_gemm_gelu(residual_norm,
                                     inter_w,
                                     inter_b,
                                     output_w,
                                     config.epsilon,
                                     config.pre_layer_norm,
                                     False)
        else:
            output, residual_add = mlp_gemm_func(input,
                                             residual,
                                             bias,
                                             inter_w,
                                             output_w,
                                             inter_b,
                                             attn_nw,
                                             attn_nb,
                                             config.epsilon,
                                             config.pre_layer_norm,
                                             config.mlp_after_attn,
                                             inter_w.scale,
                                             output_w.scale,
                                             config.quantize,
                                             config.quantization_bits,
                                             config.mlp_act_func_type)

        residual = residual if config.pre_layer_norm else residual_add
        residual_add_func(
            output,                # hidden state
            residual,              # residual
            input,                 # attention output
            bias if bias is not None else output_b,
            output_b,
            config.mp_size,         # model parallel size
            config.mlp_after_attn,  # whether mlp is after attention (GPTJ model architecture runs the MLP layer in parallel with attention)
            bias is not None,       # whether bias addition is fused
            config.pre_layer_norm)  # whether the layer norm is applied before attention
        if mp_group is not None and dist.get_world_size(group=mp_group) > 1:
            dist.all_reduce(residual, group=mp_group)
        return residual

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError('You are running with DeepSpeed Inference mode. \
                            Please switch to Training mode for running backward!')

class DeepSpeedMLP(nn.Module):
    _inter_w_buffers = []

    def __init__(self, config, mp_group=None, q_scales=None, q_groups=1, merge_count=1, mlp_extra_grouping=False):
        super(DeepSpeedMLP, self).__init__()

        self.config = config
        
        data_type = torch.int8 if config.quantize else torch.half if self.config.dtype == torch.half else torch.float
        data_type_fp = torch.half if self.config.dtype == torch.half else torch.float
        ###data_type = torch.int8 if config.quantize else torch.half if config.fp16 else torch.float
        ###data_type_fp = torch.half if config.fp16 else torch.float
        #device = get_accelerator().current_device_name()
        device = torch.cuda.current_device() if config.bigscience_bloom else 'cpu'
        proj_factor = 2 if self.config.mlp_act_func_type in GATED_ACTIVATION_TYPES else 1
        self.config.intermediate_size = self.config.intermediate_size if self.config.intermediate_size > 0 else 4 * self.config.hidden_size
        self.intm_w_sz_per_partition = self.config.intermediate_size * proj_factor // self.config.mp_size
        self.intm_o_sz_per_partition = self.config.intermediate_size // self.config.mp_size
        
        if self.config.set_empty_params:
            self.attn_nw = None
            self.attn_nb = None
            self.inter_w = None
            self.inter_b = None
            self.inter_up_w = None
            self.inter_up_b = None
            self.inter_gate_w = None
            self.inter_gate_b = None
            self.output_w = None
            self.output_b = None

        else:
            self.attn_nw = nn.Parameter(torch.empty(self.config.hidden_size,
                                                dtype=data_type_fp,
                                                device=device),
                                    requires_grad=False)
            self.attn_nb = nn.Parameter(torch.empty(self.config.hidden_size,
                                                dtype=data_type_fp,
                                                device=device),
                                    requires_grad=False)
            intm_size_per_partition = self.config.intermediate_size // self.config.mp_size
            self.inter_w = nn.Parameter(torch.empty(self.config.hidden_size,
                                                intm_size_per_partition // 2 if self.config.quantization_bits==4 else intm_size_per_partition,
                                                dtype=data_type,
                                                device=device),
                                    requires_grad=False)
            self.inter_b = nn.Parameter(torch.empty(intm_size_per_partition,
                                                dtype=data_type_fp,
                                                device=device),
                                    requires_grad=False)
            self.output_w = nn.Parameter(torch.empty(intm_size_per_partition,
                                                 self.config.hidden_size // 2 if self.config.quantization_bits==4 else self.config.hidden_size,
                                                 dtype=data_type,
                                                 device=device),
                                     requires_grad=False)
            self.output_b = nn.Parameter(torch.empty(self.config.hidden_size,
                                                 dtype=data_type_fp,
                                                 device=device),
                                     requires_grad=False)

        # used for quantization
        self.q_scales = q_scales
        self.q_groups = q_groups * 2 if mlp_extra_grouping else q_groups
        self.merge_count = int(math.log2(merge_count))
        self.mp_group = mp_group
        
        # load the cuda module
        global inference_cuda_module
        if inference_cuda_module is None:
            builder = op_builder.InferenceBuilder()
            inference_cuda_module = builder.load()

        self.bias_residual_func = inference_cuda_module.bias_residual_fp16 if self.config.dtype == torch.half or config.quantize else \
                                    inference_cuda_module.bias_residual_fp32

        self.residual_add_func = inference_cuda_module.residual_add_bias_fp16 if self.config.dtype == torch.half or config.quantize else \
                                    inference_cuda_module.residual_add_bias_fp32
        
        self.mp_group = mp_group
        self.mlp_gemm_func = inference_cuda_module.mlp_gemm_fp16 if self.config.dtype == torch.half else \
                                    inference_cuda_module.mlp_gemm_fp32
        self.vector_matmul_func = inference_cuda_module.vector_matmul_fp16 if self.config.dtype == torch.half else \
                                inference_cuda_module.vector_matmul_fp32
        self.fused_gemm_gelu = inference_cuda_module.fused_gemm_gelu_fp16 if self.config.dtype == torch.half else \
                                    inference_cuda_module.fused_gemm_gelu_fp32 

        #self.mlp_gemm_func = MLPGemmOp(config)
        #self.vector_matmul_func = VectorMatMulOp(config)
        #self.fused_gemm_gelu = GELUGemmOp(config)
        #self.residual_add_func = ResidualAddOp(config)

        if len(DeepSpeedMLP._inter_w_buffers) == 0:
            DeepSpeedMLP._inter_w_buffers = [
                torch.empty(self.config.hidden_size, self.intm_w_sz_per_partition, dtype=data_type, device=device),
                torch.empty(self.intm_w_sz_per_partition, dtype=data_type_fp, device=device)
            ]

    def _merge_inter_w(self):
        inter_w = DeepSpeedMLP._inter_w_buffers[0]
        inter_w[:self.intm_w_sz_per_partition, :] = self.inter_up_w  # type: ignore
        inter_w[self.intm_w_sz_per_partition:, :] = self.inter_gate_w  # type: ignore
        if self.inter_up_b is not None:
            inter_b = DeepSpeedMLP._inter_w_buffers[1]
            inter_b[:self.intm_w_sz_per_partition] = self.inter_up_b  # type: ignore
            inter_b[self.intm_w_sz_per_partition:] = self.inter_gate_b  # type: ignore
        return DeepSpeedMLP._inter_w_buffers


    def forward(self, input, residual, residual_norm, bias):

        if self.inter_w is None:
            self._inter_w, self._inter_b = self._merge_inter_w()
        else:
            self._inter_w = self.inter_w
            self._inter_b = self.inter_b

        residual_add = None
        if self.attn_nw is None:
            output = self.fused_gemm_gelu(input=residual_norm,
                                          weight=self.inter_w,
                                          bias=self.inter_b,
                                          weight_out=self.output_w)
        else:
            output, residual_add = self.mlp_gemm_func(input=input,
                                                      residual=residual,
                                                      weight_interm=self.inter_w,
                                                      weight_out=self.output_w,
                                                      input_bias=bias,
                                                      bias=self.inter_b,
                                                      gamma=self.attn_nw,
                                                      beta=self.attn_nb)

        residual = self.residual_add_func(hidden_state=output,
                                          residual=residual,
                                          add_bias=bias is not None,
                                          attention_output=input,
                                          attention_bias=bias if bias is not None else self.output_b,
                                          final_bias=self.output_b,
                                          residual_add=residual_add)
        if self.mp_group is not None and dist.get_world_size(group=self.mp_group) > 1:
            dist.all_reduce(residual, group=self.mp_group)

        return residual
