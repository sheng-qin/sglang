from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from compressed_tensors.quantization import QuantizationStrategy

from sglang.srt.hardware_backend.npu.quantization.fused_moe_method_npu import (
    NPUW8A8Int8DynamicMoEMethod,
)
from sglang.srt.layers.moe import MoeRunnerConfig
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsMoEScheme,
)
from sglang.srt.utils import set_weight_attrs

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )

__all__ = [
    "NPUCompressedTensorsW8A8Int8DynamicMoE",
    "IxformerCompressedTensorsW8A8Int8DynamicMoE",
]

logger = logging.getLogger(__name__)


class NPUCompressedTensorsW8A8Int8DynamicMoE(CompressedTensorsMoEScheme):

    def __init__(self, weight_quant, input_quant):
        self.weight_quant = weight_quant
        self.input_quant = input_quant
        self.kernel = NPUW8A8Int8DynamicMoEMethod()

        self.static_input_scales = not self.input_quant.dynamic
        per_channel = (
            self.weight_quant.strategy == QuantizationStrategy.CHANNEL
            and self.input_quant.strategy == QuantizationStrategy.TOKEN
        )
        if not per_channel:
            raise ValueError(
                "For INT8 Fused MoE layers, we require channelwise, "
                "dynamic per token quantization. Found "
                f"{self.weight_quant}, {self.input_quant}"
            )

        self.static_input_scales = not self.input_quant.dynamic
        if self.static_input_scales:
            raise ValueError(
                "For INT8 Fused MoE layers, we require channelwise, "
                "dynamic per token quantization. Found static input scales."
            )

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):

        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        params_dtype = torch.int8

        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        assert self.weight_quant.strategy == QuantizationStrategy.CHANNEL
        w13_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts, 2 * intermediate_size_per_partition, 1, dtype=torch.float32
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, hidden_size, 1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        # Add PER-CHANNEL quantization for FusedMoE.weight_loader.
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
        )
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # INPUT_SCALES
        assert not self.static_input_scales
        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.kernel.process_weights_after_loading(layer)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config

    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:

        return self.kernel.apply(layer, dispatch_output)


class IxformerCompressedTensorsW8A8Int8DynamicMoE(CompressedTensorsMoEScheme):
    """Compressed-tensors W8A8 INT8 fused-MoE on the Iluvatar/ixformer backend.

    Per-channel static int8 expert weights, per-token dynamic symmetric int8
    activations. The experts run via ixformer's int8 group-GEMM kernels.
    """

    def __init__(self, weight_quant, input_quant):
        self.weight_quant = weight_quant
        self.input_quant = input_quant

        per_channel = (
            self.weight_quant.strategy == QuantizationStrategy.CHANNEL
            and self.input_quant.strategy == QuantizationStrategy.TOKEN
        )
        if not per_channel:
            raise ValueError(
                "For INT8 Fused MoE layers, we require channelwise, "
                "dynamic per token quantization. Found "
                f"{self.weight_quant}, {self.input_quant}"
            )

        self.static_input_scales = not self.input_quant.dynamic
        if self.static_input_scales:
            raise ValueError(
                "For INT8 Fused MoE layers, we require channelwise, "
                "dynamic per token quantization. Found static input scales."
            )

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        params_dtype = torch.int8

        # WEIGHTS: w13 = (E, 2I, H), w2 = (E, H, I), int8 (per-expert [N, K]).
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES: per-output-channel fp32.
        assert self.weight_quant.strategy == QuantizationStrategy.CHANNEL
        w13_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts, 2 * intermediate_size_per_partition, 1, dtype=torch.float32
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, hidden_size, 1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
        )
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # INPUT_SCALES: dynamic per-token, so no static input scale.
        assert not self.static_input_scales
        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # ixformer's moe_w8a8_group_gemm consumes the per-expert weights as
        # (E, N, K) int8 with format="TN" and the channel scales as-is, so no
        # transpose/reshape is needed here.
        pass

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config

    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        import ixformer.inference.functions as ixf

        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        assert self.moe_runner_config.activation == "silu", (
            f"activation = {self.moe_runner_config.activation} is not supported "
            "by the ixformer W8A8 MoE path."
        )

        x = dispatch_output.hidden_states
        topk_weights, topk_ids, router_logits = dispatch_output.topk_output

        num_tokens = x.shape[0]
        top_k = topk_ids.shape[-1]
        num_experts = router_logits.shape[-1]
        dtype = x.dtype
        scaling_factor = (
            self.moe_runner_config.routed_scaling_factor
            if self.moe_runner_config.routed_scaling_factor is not None
            else 1.0
        )
        expand_tokens = num_tokens * top_k

        src_to_dst, sorted_token_ids, expert_sizes_gpu, _ = (
            ixf.moe_compute_token_index(
                topk_ids=topk_ids.to(torch.int32),
                num_experts=num_experts,
            )
        )
        tokens_per_experts = expert_sizes_gpu.cpu()

        # expand + reorder + dynamic per-token int8 quant of activations.
        i8_hidden, a_scale = ixf.moe_expand_input_dynamic_scaled_int8(
            hidden_states=x,
            dst_to_src=sorted_token_ids,
            dst_tokens=expand_tokens,
            topk=top_k,
            src_to_dst=src_to_dst,
            topk_ids=None,
            smooth_scales=layer.w13_input_scale,
        )

        # group GEMM 1 (gate_up); input already expert-ordered -> dst_to_src=None.
        gate_up = ixf.moe_w8a8_group_gemm(
            input=i8_hidden,
            weight=layer.w13_weight,
            i_scales=a_scale,
            w_scales=layer.w13_weight_scale,
            output_dtype=dtype,
            tokens_per_experts=tokens_per_experts,
            dst_to_src=None,
            format="TN",
        )

        # SwiGLU + dynamic per-token int8 requant of the intermediate.
        i8_act, a2_scale = ixf.activation_dynamic_scaled_int8(
            input=gate_up,
            bias=None,
            smooth_scales=layer.w2_input_scale,
            dst_to_src=sorted_token_ids,
            topk_ids=None,
            act_type="swiglu",
        )

        # group GEMM 2 (down); scatter back to token order via dst_to_src.
        down = ixf.moe_w8a8_group_gemm(
            input=i8_act,
            weight=layer.w2_weight,
            i_scales=a2_scale,
            w_scales=layer.w2_weight_scale,
            output_dtype=dtype,
            tokens_per_experts=tokens_per_experts,
            dst_to_src=sorted_token_ids,
            format="TN",
        )

        output = ixf.moe_output_reduce_sum(
            input=down.view(num_tokens, top_k, -1),
            topk_weight=topk_weights,
            scaling_factor=scaling_factor,
        )
        return StandardCombineInput(hidden_states=output)
