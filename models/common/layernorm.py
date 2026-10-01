# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.utility_functions import copy_to_buffer
from models.tt_transformers.tt.common import Mode

TILE = 32
SHARD_HEIGHT = TILE  # Current ttnn.layer_norm implementation requires shard height to be a single tile


class LayerNorm(LightweightModule):
    """
    Mean-centred LayerNorm without bias, as used by Cohere2.

    Same interface as models.common.rmsnorm.RMSNorm so DistributedNorm wraps either.
    """

    def __init__(
        self,
        device,
        dim,
        state_dict,
        weight_key,
        layer_num=None,
        state_dict_prefix=None,
        weight_cache_path=None,
        weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        weight_dtype=ttnn.bfloat16,
        is_distributed=None,
        eps: float = 1e-05,
        add_unit_offset=False,
        sharded_program_config=None,
        sharded_output_config=None,
        output_mem_config=None,
        ccl_topology=ttnn.Topology.Ring,
        tt_ccl=None,
        fp32_dest_acc_en=True,
    ):
        super().__init__()
        self.device = device
        self.eps: float = eps
        self.is_distributed = is_distributed
        self.ccl_topology = ccl_topology
        self.tt_ccl = tt_ccl
        self.add_unit_offset = add_unit_offset

        if state_dict_prefix:
            weight_name = f"{state_dict_prefix}{weight_key}.weight"
        else:
            if layer_num is None:
                weight_name = f"{weight_key}.weight"
            else:
                weight_name = f"layers.{layer_num}.{weight_key}.weight"

        torch_weight = (
            state_dict[weight_name].unsqueeze(0).view(1, 1, dim).reshape([1, 1, dim // SHARD_HEIGHT, SHARD_HEIGHT])
        )

        # Add offset before caching
        if add_unit_offset:
            torch_weight = torch_weight + 1.0

        # Compatibility with models that don't use mesh devices (e.g. single-chip Mistral-7b)
        is_mesh_device = device.__class__.__name__ == "MeshDevice"

        self.weight = ttnn.as_tensor(
            torch_weight,
            device=device,
            dtype=weight_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=weight_memory_config,
            cache_file_name=None if weight_cache_path is None else weight_cache_path / weight_name,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
        )

        if self.is_distributed:
            self.weight_distributed = ttnn.as_tensor(
                torch_weight,
                device=device,
                dtype=weight_dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=weight_memory_config,
                cache_file_name=(
                    None if weight_cache_path is None else weight_cache_path / (weight_name + "_distributed")
                ),
                mesh_mapper=(
                    ttnn.ShardTensor2dMesh(device, dims=(None, 2), mesh_shape=list(device.shape))
                    if is_mesh_device
                    else None
                ),
            )

        self.sharded_output_config = sharded_output_config
        self.sharded_program_config = sharded_program_config
        self.output_mem_config = output_mem_config

        self.compute_kernel_config_hifi2 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=fp32_dest_acc_en,
            packer_l1_acc=True,
        )

    def update(self, *, weight: ttnn.Tensor) -> None:
        """In-place replace the LayerNorm gamma via ``ttnn.copy``. See RMSNorm.update."""
        assert not self.add_unit_offset, "LayerNorm.update does not support add_unit_offset=True"
        copy_to_buffer(weight, self.weight, self.weight.dtype)

        if getattr(self, "weight_distributed", None) is not None:
            partitioned = ttnn.mesh_partition(
                self.weight,
                memory_config=self.weight_distributed.memory_config(),
                dim=2,
                cluster_axis=1,
            )
            copy_to_buffer(partitioned, self.weight_distributed, self.weight_distributed.dtype)

    def forward(
        self,
        x: ttnn.Tensor,
        mode: Mode | str,
        in_sharded=False,
        out_sharded=False,
        norm_config=None,
    ) -> ttnn.Tensor:
        if isinstance(mode, str):
            try:
                mode = Mode(mode)
            except ValueError:
                raise ValueError(f"Invalid mode: {mode}")
        elif not isinstance(mode, Mode):
            raise ValueError(f"Invalid mode: {mode}")

        sharded_program_config = norm_config.get("sharded_program_config") if norm_config else None
        sharded_output_config = norm_config.get("sharded_output_config") if norm_config else None
        output_mem_config = norm_config.get("output_mem_config") if norm_config else None
        # Optional L1 placement for the distributed 3-op outputs (pre/gather/post); None -> DRAM default.
        distributed_out_mc = norm_config.get("distributed_output_mem_config") if norm_config else None

        # If input is sharded do sharded LayerNorm and optionally return sharded output
        program_config = sharded_program_config if in_sharded else None
        memory_config = sharded_output_config if out_sharded else None
        distributed = self.is_distributed and self.is_distributed(mode)
        weight = self.weight_distributed if distributed else self.weight

        if in_sharded:
            assert not distributed, "Distributed LayerNorm does not support sharded inputs"
        else:
            assert not out_sharded, "Non-sharded version of LayerNorm cannot output a sharded tensor"

        if distributed:
            x = self._distributed_layer_norm(
                x,
                epsilon=self.eps,
                weight=weight,
                compute_kernel_config=self.compute_kernel_config_hifi2,
                output_memory_config=distributed_out_mc,
            )
        else:
            x = ttnn.layer_norm(
                x,
                epsilon=self.eps,
                weight=weight,
                bias=None,
                program_config=program_config,
                memory_config=memory_config,
                compute_kernel_config=self.compute_kernel_config_hifi2,
            )

        if in_sharded and not out_sharded:
            return ttnn.sharded_to_interleaved(x)
        else:
            if output_mem_config is not None:
                x = ttnn.to_memory_config(x, output_mem_config)
            return x

    def _distributed_layer_norm(
        self,
        inp,
        epsilon=None,
        weight=None,
        program_config=None,
        memory_config=None,
        compute_kernel_config=None,
        output_memory_config=None,
    ):
        assert program_config is None, "Distributed LayerNorm does not support sharded inputs"
        assert memory_config is None, "Distributed LayerNorm does not support sharded outputs"
        assert self.tt_ccl is not None, "Distributed LayerNorm requires tt_ccl"

        # Interleaved output placement for the 3 ops; default DRAM (matches the prior hardcoded behavior).
        mc = output_memory_config if output_memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

        # Run distributed layernorm part 1
        tt_stats = ttnn.layer_norm_pre_all_gather(
            inp, compute_kernel_config=compute_kernel_config, dtype=ttnn.bfloat16, memory_config=mc
        )
        # AllGather stats
        tt_stats = ttnn.experimental.all_gather_async(
            tt_stats,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
            num_links=1,
            topology=self.ccl_topology,
            memory_config=mc,
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )
        # Run distributed layernorm part 2
        tt_out = ttnn.layer_norm_post_all_gather(
            inp,
            tt_stats,
            epsilon=epsilon,
            weight=weight,
            compute_kernel_config=compute_kernel_config,
            memory_config=mc,
        )
        tt_stats.deallocate(True)

        return tt_out
