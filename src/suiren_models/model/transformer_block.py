from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch_geometric
import numpy as np
import os

import e3nn.o3 as o3
import torch.nn.functional as F

from .activation import (
    GateActivation,
    S2Activation,
    SeparableS2Activation,
    SmoothLeakyReLU,
)
from .drop import EquivariantDropoutArraySphericalHarmonics, GraphDropPath
from .layer_norm import get_normalization_layer
from .radial_function import RadialFunction
from .so2_ops import SO2_Convolution
from .so3 import SO3_Embedding, SO3_LinearV2

_Jd = torch.load(os.path.join(os.path.dirname(__file__), "Jd.pt"))

class SO2EquivariantGraphAttention(torch.nn.Module):
    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        num_heads: int,
        attn_alpha_channels: int,
        attn_value_channels: int,
        output_channels: int,
        lmax_list: list[int],
        mmax_list: list[int],
        SO3_rotation,
        mappingReduced,
        SO3_grid,
        max_num_elements: int,
        edge_channels_list,
        use_atom_edge_embedding: bool = True,
        use_m_share_rad: bool = False,
        activation="scaled_silu",
        use_s2_act_attn: bool = False,
        use_attn_renorm: bool = True,
        use_gate_act: bool = False,
        use_sep_s2_act: bool = True,
        alpha_drop: float = 0.0,
    ):
        super().__init__()

        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.attn_alpha_channels = attn_alpha_channels
        self.attn_value_channels = attn_value_channels
        self.output_channels = output_channels
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.num_resolutions = len(self.lmax_list)

        self.SO3_rotation = SO3_rotation
        self.mappingReduced = mappingReduced
        self.SO3_grid_up = SO3_grid

        # Create edge scalar (invariant to rotations) features
        # Embedding function of the atomic numbers
        self.max_num_elements = max_num_elements
        self.edge_channels_list = copy.deepcopy(edge_channels_list)
        self.use_atom_edge_embedding = use_atom_edge_embedding
        self.use_m_share_rad = use_m_share_rad

        if self.use_atom_edge_embedding:
            self.source_embedding = nn.Embedding(
                self.max_num_elements, self.edge_channels_list[-1]
            )
            self.target_embedding = nn.Embedding(
                self.max_num_elements, self.edge_channels_list[-1]
            )
            nn.init.uniform_(self.source_embedding.weight.data, -0.001, 0.001)
            nn.init.uniform_(self.target_embedding.weight.data, -0.001, 0.001)
            self.edge_channels_list[0] = (
                self.edge_channels_list[0] + 2 * self.edge_channels_list[-1]
            )
        else:
            self.source_embedding, self.target_embedding = None, None

        self.use_s2_act_attn = use_s2_act_attn
        self.use_attn_renorm = use_attn_renorm
        self.use_gate_act = use_gate_act
        self.use_sep_s2_act = use_sep_s2_act

        assert not self.use_s2_act_attn  # since this is not used

        # Create SO(2) convolution blocks
        extra_m0_output_channels = None
        if not self.use_s2_act_attn:
            extra_m0_output_channels = self.num_heads * self.attn_alpha_channels
            if self.use_gate_act:
                extra_m0_output_channels = (
                    extra_m0_output_channels
                    + max(self.lmax_list) * self.hidden_channels
                )
            else:
                if self.use_sep_s2_act:
                    extra_m0_output_channels = (
                        extra_m0_output_channels + self.hidden_channels
                    )

        if self.use_m_share_rad:
            self.edge_channels_list = [
                *self.edge_channels_list,
                2 * self.sphere_channels * (max(self.lmax_list) + 1),
            ]
            self.rad_func = RadialFunction(self.edge_channels_list)
            expand_index = torch.zeros([(max(self.lmax_list) + 1) ** 2]).long()
            for lval in range(max(self.lmax_list) + 1):
                start_idx = lval**2
                length = 2 * lval + 1
                expand_index[start_idx : (start_idx + length)] = lval
            self.register_buffer("expand_index", expand_index)

        self.so2_conv_1 = SO2_Convolution(
            2 * self.sphere_channels,
            self.hidden_channels,
            self.lmax_list,
            self.mmax_list,
            self.mappingReduced,
            internal_weights=(bool(self.use_m_share_rad)),
            edge_channels_list=(
                self.edge_channels_list if not self.use_m_share_rad else None
            ),
            extra_m0_output_channels=extra_m0_output_channels,  # for attention weights and/or gate activation
        )

        if self.use_s2_act_attn:
            self.alpha_norm = None
            self.alpha_act = None
            self.alpha_dot = None
        else:
            if self.use_attn_renorm:
                self.alpha_norm = torch.nn.LayerNorm(self.attn_alpha_channels)
            else:
                self.alpha_norm = torch.nn.Identity()
            self.alpha_act = SmoothLeakyReLU()
            self.alpha_dot = torch.nn.Parameter(
                torch.randn(self.num_heads, self.attn_alpha_channels)
            )
            # torch_geometric.nn.inits.glorot(self.alpha_dot) # Following GATv2
            std = 1.0 / math.sqrt(self.attn_alpha_channels)
            torch.nn.init.uniform_(self.alpha_dot, -std, std)

        self.alpha_dropout = None
        if alpha_drop != 0.0:
            self.alpha_dropout = torch.nn.Dropout(alpha_drop)

        if self.use_gate_act:
            self.gate_act = GateActivation(
                lmax=max(self.lmax_list),
                mmax=max(self.mmax_list),
                num_channels=self.hidden_channels,
            )
        else:
            if self.use_sep_s2_act:
                # separable S2 activation
                self.s2_act = SeparableS2Activation(
                    lmax=max(self.lmax_list), mmax=max(self.mmax_list)
                )
            else:
                # S2 activation
                self.s2_act = S2Activation(
                    lmax=max(self.lmax_list), mmax=max(self.mmax_list)
                )

        self.so2_conv_2 = SO2_Convolution(
            self.hidden_channels,
            self.num_heads * self.attn_value_channels,
            self.lmax_list,
            self.mmax_list,
            self.mappingReduced,
            internal_weights=True,
            edge_channels_list=None,
            extra_m0_output_channels=(
                self.num_heads if self.use_s2_act_attn else None
            ),  # for attention weights
        )

        self.proj = SO3_LinearV2(
            self.num_heads * self.attn_value_channels,
            self.output_channels,
            lmax=self.lmax_list[0],
        )

    def forward(
        self,
        x: torch.Tensor,
        atomic_numbers,
        edge_distance: torch.Tensor,
        edge_index,
        node_offset: int = 0,
        ssp: bool = False,
    ):
        if ssp:
            task_idx = 1
        else:
            task_idx = 0
        # Compute edge scalar features (invariant to rotations)
        # Uses atomic numbers and edge distance as inputs
        if self.use_atom_edge_embedding:
            source_element = atomic_numbers[edge_index[0]]  # Source atom atomic number
            target_element = atomic_numbers[edge_index[1]]  # Target atom atomic number
            source_embedding = self.source_embedding(source_element)
            target_embedding = self.target_embedding(target_element)
            x_edge = torch.cat(
                (edge_distance, source_embedding, target_embedding), dim=1
            )
        else:
            x_edge = edge_distance

        x_source = x.clone()
        x_target = x.clone()
        x_source._expand_edge(edge_index[0, :])
        x_target._expand_edge(edge_index[1, :])

        x_message_data = torch.cat((x_source.embedding, x_target.embedding), dim=2)
        x_message = SO3_Embedding(
            0,
            x_target.lmax_list.copy(),
            x_target.num_channels * 2,
            device=x_target.device,
            dtype=x_target.dtype,
        )
        x_message.set_embedding(x_message_data)
        x_message.set_lmax_mmax(self.lmax_list.copy(), self.mmax_list.copy())

        # radial function (scale all m components within a type-L vector of one channel with the same weight)
        if self.use_m_share_rad:
            x_edge_weight = self.rad_func(x_edge)
            x_edge_weight = x_edge_weight.reshape(
                -1, (max(self.lmax_list) + 1), 2 * self.sphere_channels
            )
            x_edge_weight = torch.index_select(
                x_edge_weight, dim=1, index=self.expand_index
            )  # [E, (L_max + 1) ** 2, C]
            x_message.embedding = x_message.embedding * x_edge_weight

        # Rotate the irreps to align with the edge
        x_message._rotate(self.SO3_rotation[task_idx], self.lmax_list, self.mmax_list)

        # First SO(2)-convolution
        if self.use_s2_act_attn:
            x_message = self.so2_conv_1(x_message, x_edge)
        else:
            x_message, x_0_extra = self.so2_conv_1(x_message, x_edge)

        # Activation
        x_alpha_num_channels = self.num_heads * self.attn_alpha_channels
        if self.use_gate_act:
            # Gate activation
            x_0_gating = x_0_extra.narrow(
                1,
                x_alpha_num_channels,
                x_0_extra.shape[1] - x_alpha_num_channels,
            )  # for activation
            x_0_alpha = x_0_extra.narrow(
                1, 0, x_alpha_num_channels
            )  # for attention weights
            x_message.embedding = self.gate_act(x_0_gating, x_message.embedding)
        else:
            if self.use_sep_s2_act:
                x_0_gating = x_0_extra.narrow(
                    1,
                    x_alpha_num_channels,
                    x_0_extra.shape[1] - x_alpha_num_channels,
                )  # for activation
                x_0_alpha = x_0_extra.narrow(
                    1, 0, x_alpha_num_channels
                )  # for attention weights
                x_message.embedding = self.s2_act(
                    x_0_gating, x_message.embedding, self.SO3_grid_up
                )
            else:
                x_0_alpha = x_0_extra
                x_message.embedding = self.s2_act(x_message.embedding, self.SO3_grid_up)
            # x_message._grid_act(self.SO3_grid, self.value_act, self.mappingReduced)

        # Second SO(2)-convolution
        if self.use_s2_act_attn:
            x_message, x_0_extra = self.so2_conv_2(x_message, x_edge)
        else:
            x_message = self.so2_conv_2(x_message, x_edge)

        # Attention weights
        if self.use_s2_act_attn:
            alpha = x_0_extra
        else:
            x_0_alpha = x_0_alpha.reshape(-1, self.num_heads, self.attn_alpha_channels)
            x_0_alpha = self.alpha_norm(x_0_alpha)
            x_0_alpha = self.alpha_act(x_0_alpha)
            alpha = torch.einsum("bik, ik -> bi", x_0_alpha, self.alpha_dot)
        alpha = torch_geometric.utils.softmax(alpha, edge_index[1])
        alpha = alpha.reshape(alpha.shape[0], 1, self.num_heads, 1)
        if self.alpha_dropout is not None:
            alpha = self.alpha_dropout(alpha)

        # Attention weights * non-linear messages
        attn = x_message.embedding
        attn = attn.reshape(
            attn.shape[0],
            attn.shape[1],
            self.num_heads,
            self.attn_value_channels,
        )
        attn = attn * alpha
        attn = attn.reshape(
            attn.shape[0],
            attn.shape[1],
            self.num_heads * self.attn_value_channels,
        )
        x_message.embedding = attn

        # Rotate back the irreps
        x_message._rotate_inv(self.SO3_rotation[task_idx], self.mappingReduced)

        # Compute the sum of the incoming neighboring messages for each target node
        x_message._reduce_edge(edge_index[1] - node_offset, len(x.embedding))

        # Project
        return self.proj(x_message)


class FeedForwardNetwork(torch.nn.Module):
    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        output_channels: int,
        lmax_list: list[int],
        mmax_list: list[int],
        SO3_grid,
        activation: str = "scaled_silu",
        use_gate_act: bool = False,
        use_grid_mlp: bool = False,
        use_sep_s2_act: bool = True,
        norm_type='layer_norm_sh',
        norm_flag: bool = True,
    ):
        super().__init__()
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.num_resolutions = len(lmax_list)
        self.sphere_channels_all = self.num_resolutions * self.sphere_channels
        self.SO3_grid_up = SO3_grid
        self.use_gate_act = use_gate_act
        self.use_grid_mlp = use_grid_mlp
        self.use_sep_s2_act = use_sep_s2_act

        self.max_lmax = max(self.lmax_list)

        if norm_type is None:
            norm_type = 'layer_norm_sh'
        self.norm_flag = norm_flag
        if self.norm_flag:
            self.norm = get_normalization_layer(norm_type, lmax=self.max_lmax, num_channels=sphere_channels)

        self.so3_linear_1 = SO3_LinearV2(
            self.sphere_channels_all, self.hidden_channels, lmax=self.max_lmax
        )
        if self.use_grid_mlp:
            if self.use_sep_s2_act:
                self.scalar_mlp = nn.Sequential(
                    nn.Linear(
                        self.sphere_channels_all,
                        self.hidden_channels,
                        bias=True,
                    ),
                    nn.SiLU(),
                )
            else:
                self.scalar_mlp = None
            self.grid_mlp = nn.Sequential(
                nn.Linear(self.hidden_channels, 2 * self.hidden_channels, bias=False),
                nn.SiLU(),
                nn.Linear(2 * self.hidden_channels, 2 * self.hidden_channels, bias=False),
                nn.SiLU(),
                nn.Linear(2 * self.hidden_channels, self.hidden_channels, bias=False),
            )
        else:
            if self.use_gate_act:
                self.gating_linear = torch.nn.Linear(
                    self.sphere_channels_all,
                    self.max_lmax * self.hidden_channels,
                )
                self.gate_act = GateActivation(
                    self.max_lmax, self.max_lmax, self.hidden_channels
                )
            else:
                if self.use_sep_s2_act:
                    self.gating_linear = torch.nn.Linear(
                        self.sphere_channels_all, self.hidden_channels
                    )
                    self.s2_act = SeparableS2Activation(self.max_lmax, self.max_lmax)
                else:
                    self.gating_linear = None
                    self.s2_act = S2Activation(self.max_lmax, self.max_lmax)
        self.so3_linear_2 = SO3_LinearV2(
            self.hidden_channels, self.output_channels, lmax=self.max_lmax
        )

    def forward(self, ori_input_embedding):
        input_embedding = ori_input_embedding.clone()
        if self.norm_flag:
            input_embedding.embedding = self.norm(input_embedding.embedding)
        gating_scalars = None
        if self.use_grid_mlp:
            if self.use_sep_s2_act:
                gating_scalars = self.scalar_mlp(
                    input_embedding.embedding.narrow(1, 0, 1)
                )
        else:
            if self.gating_linear is not None:
                gating_scalars = self.gating_linear(
                    input_embedding.embedding.narrow(1, 0, 1)
                )

        input_embedding = self.so3_linear_1(input_embedding)

        if self.use_grid_mlp:
            # Project to grid
            input_embedding_grid = input_embedding.to_grid(
                self.SO3_grid_up, lmax=self.max_lmax
            )
            # Perform point-wise operations
            input_embedding_grid = self.grid_mlp(input_embedding_grid)
            # Project back to spherical harmonic coefficients
            input_embedding._from_grid(
                input_embedding_grid, self.SO3_grid_up, lmax=self.max_lmax
            )

            if self.use_sep_s2_act:
                input_embedding.embedding = torch.cat(
                    (
                        gating_scalars,
                        input_embedding.embedding.narrow(
                            1, 1, input_embedding.embedding.shape[1] - 1
                        ),
                    ),
                    dim=1,
                )
        else:
            if self.use_gate_act:
                input_embedding.embedding = self.gate_act(
                    gating_scalars, input_embedding.embedding
                )
            else:
                if self.use_sep_s2_act:
                    input_embedding.embedding = self.s2_act(
                        gating_scalars,
                        input_embedding.embedding,
                        self.SO3_grid_up,
                    )
                else:
                    input_embedding.embedding = self.s2_act(
                        input_embedding.embedding, self.SO3_grid_up
                    )

        return self.so3_linear_2(input_embedding)


class TransBlockV2(torch.nn.Module):
    def __init__(
        self,
        sphere_channels: int,
        attn_hidden_channels: int,
        num_heads: int,
        attn_alpha_channels: int,
        attn_value_channels: int,
        ffn_hidden_channels: int,
        output_channels: int,
        lmax_list: list[int],
        mmax_list: list[int],
        SO3_rotation,
        mappingReduced,
        SO3_grid,
        max_num_elements: int,
        edge_channels_list: list[int],
        use_atom_edge_embedding: bool = True,
        use_m_share_rad: bool = False,
        attn_activation: str = "silu",
        use_s2_act_attn: bool = False,
        use_attn_renorm: bool = True,
        ffn_activation: str = "silu",
        use_gate_act: bool = False,
        use_grid_mlp: bool = False,
        use_sep_s2_act: bool = True,
        norm_type: str = "rms_norm_sh",
        alpha_drop: float = 0.0,
        drop_path_rate: float = 0.0,
        proj_drop: float = 0.0,
        num_experts_steerable: int = 1,
        num_experts_spherical: int = 1,
        tos2grid=None,
    ) -> None:
        super().__init__()

        max_lmax = max(lmax_list)
        self.norm_1 = get_normalization_layer(
            norm_type, lmax=max_lmax, num_channels=sphere_channels
        )

        self.ga = SO2EquivariantGraphAttention(
            sphere_channels=sphere_channels,
            hidden_channels=attn_hidden_channels,
            num_heads=num_heads,
            attn_alpha_channels=attn_alpha_channels,
            attn_value_channels=attn_value_channels,
            output_channels=sphere_channels,
            lmax_list=lmax_list,
            mmax_list=mmax_list,
            SO3_rotation=SO3_rotation,
            mappingReduced=mappingReduced,
            SO3_grid=SO3_grid,
            max_num_elements=max_num_elements,
            edge_channels_list=edge_channels_list,
            use_atom_edge_embedding=use_atom_edge_embedding,
            use_m_share_rad=use_m_share_rad,
            activation=attn_activation,
            use_s2_act_attn=use_s2_act_attn,
            use_attn_renorm=use_attn_renorm,
            use_gate_act=use_gate_act,
            use_sep_s2_act=use_sep_s2_act,
            alpha_drop=alpha_drop,
        )

        self.drop_path = GraphDropPath(drop_path_rate) if drop_path_rate > 0.0 else None
        self.proj_drop = (
            EquivariantDropoutArraySphericalHarmonics(proj_drop, drop_graph=False)
            if proj_drop > 0.0
            else None
        )

        # self.norm_2 = get_normalization_layer(
        #     norm_type, lmax=max_lmax, num_channels=sphere_channels
        # )
        self.norm_gate = nn.LayerNorm(sphere_channels)

        # EST Module
        self.num_experts_steerable = num_experts_steerable
        self.num_experts_spherical = num_experts_spherical
        self.num_experts = self.num_experts_steerable + self.num_experts_spherical

        self.gate_node = nn.Sequential(nn.Linear(sphere_channels, sphere_channels),
            nn.SiLU(),
            nn.Linear(sphere_channels, self.num_experts),
            nn.SiLU())

        self.tos2grid = tos2grid
        self.est = EST_module(sphere_channels, max_lmax, self.num_experts_spherical, self.tos2grid)

        self.ffn = torch.nn.ModuleList([
            FeedForwardNetwork(
            sphere_channels=sphere_channels,
            hidden_channels=ffn_hidden_channels, 
            output_channels=output_channels,
            lmax_list=lmax_list,
            mmax_list=mmax_list,
            SO3_grid=SO3_grid,  
            activation=ffn_activation,
            use_gate_act=use_gate_act,
            use_grid_mlp=use_grid_mlp,
            use_sep_s2_act=use_sep_s2_act,
            norm_type=norm_type,
            )
            for _ in range(self.num_experts_steerable)
        ])

        if sphere_channels != output_channels:
            self.ffn_shortcut = SO3_LinearV2(
                sphere_channels, output_channels, lmax=max_lmax
            )
        else:
            self.ffn_shortcut = None

    def forward(
        self,
        x,  # SO3_Embedding
        atomic_numbers,
        edge_distance,
        edge_index,
        batch,  # for GraphDropPath
        node_offset: int = 0,
        ssp: bool = False,
    ):
        output_embedding = x.clone()

        x_res = output_embedding.embedding
        output_embedding.embedding = self.norm_1(output_embedding.embedding)
        output_embedding = self.ga(
            output_embedding, atomic_numbers, edge_distance, edge_index, node_offset, ssp
        )

        if self.drop_path is not None:
            output_embedding.embedding = self.drop_path(
                output_embedding.embedding, batch
            )
        if self.proj_drop is not None:
            output_embedding.embedding = self.proj_drop(
                output_embedding.embedding, batch
            )

        output_embedding.embedding = output_embedding.embedding + x_res

        x_res = output_embedding.embedding

        # EST Module
        # gate for MoE
        gate_logit = self.gate_node(self.norm_gate(output_embedding.embedding[:, 0, :]))
        gate_logit = torch.softmax(gate_logit, dim=-1)
        gate_logit_streerable, gate_logit_spherical = gate_logit[:, :self.num_experts_steerable], gate_logit[:, self.num_experts_steerable:]

        # est module
        weighted_spatial_output = self.est(output_embedding.embedding, gate_logit_spherical, batch)

        weighted_steerable_output = torch.zeros_like(output_embedding.embedding)
        for i in range(self.num_experts_steerable):
            weighted_steerable_output = weighted_steerable_output + self.ffn[i](output_embedding).embedding * gate_logit_streerable[:, i].view(-1, 1, 1)

        output_embedding.embedding = weighted_steerable_output + weighted_spatial_output

        if self.drop_path is not None:
            output_embedding.embedding = self.drop_path(
                output_embedding.embedding, batch
            )
        if self.proj_drop is not None:
            output_embedding.embedding = self.proj_drop(
                output_embedding.embedding, batch
            )

        if self.ffn_shortcut is not None:
            shortcut_embedding = SO3_Embedding(
                0,
                output_embedding.lmax_list.copy(),
                self.ffn_shortcut.in_features,
                device=output_embedding.device,
                dtype=output_embedding.dtype,
            )
            shortcut_embedding.set_embedding(x_res)
            shortcut_embedding.set_lmax_mmax(
                output_embedding.lmax_list.copy(),
                output_embedding.lmax_list.copy(),
            )
            shortcut_embedding = self.ffn_shortcut(shortcut_embedding)
            x_res = shortcut_embedding.embedding

        output_embedding.embedding = output_embedding.embedding + x_res

        return output_embedding

class EST_module(nn.Module):
    def __init__(self, dim_in, l_max, num_experts, tos2grid):
        super().__init__()
        self.dim_in = dim_in
        self.reduce_channel = self.dim_in
        self.ln1 = nn.LayerNorm(self.dim_in)
        self.W_Q = nn.Linear(self.dim_in, self.reduce_channel)
        self.W_K = nn.Linear(self.dim_in, self.reduce_channel)
        self.W_V = nn.Linear(self.dim_in, self.reduce_channel)

        self.output = torch.nn.ModuleList([
            nn.Sequential(nn.LayerNorm(self.reduce_channel),
            nn.Linear(self.dim_in, self.dim_in * 4),
            nn.SiLU(),
            nn.Linear(self.dim_in * 4, self.dim_in))
            for _ in range(num_experts)
        ])

        self.norm_l0 = nn.LayerNorm(self.dim_in)
        self.fc_input_l0 = nn.Linear(self.dim_in, self.reduce_channel)
        
        self.to_grid = tos2grid
        self.num_experts = num_experts

        self.act = nn.SiLU()

    def forward(self, message, gate_output, batch):
        
        # message (l=0)
        message_l0 = self.norm_l0(message[:, 0:1, :])
        message_l0 = self.act(self.fc_input_l0(message_l0))

        if self.training:
            last_rotation_matrix = random_rotate(batch)
        else:
            last_rotation_matrix = None
        
        # message
        message = self.to_grid.ToGrid(message, last_rotation_matrix=last_rotation_matrix)
        edge_num, s, c = message.shape
        message = message.view(edge_num, s, -1)
        message = self.ln1(message)
        q = self.W_Q(message)
        k = self.W_K(message)
        v = self.W_V(message)
        
        # add the relative orientation embedding (roe) for each sampling point
        q = self.spherical_roe(q, last_rotation_matrix=last_rotation_matrix)
        k = self.spherical_roe(k, last_rotation_matrix=last_rotation_matrix)
        
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.view(edge_num, s, -1)

        spherical_out = torch.zeros_like(out)
        for i in range(self.num_experts):
            spherical_out = spherical_out + self.output[i](out) * gate_output[:, i].view(-1, 1, 1)

        spherical_out = self.to_grid.FromGrid(spherical_out, last_rotation_matrix=last_rotation_matrix)
        spherical_out = torch.cat([message_l0, spherical_out[:, 1:, :]], dim=1)
        return spherical_out
    
    def spherical_roe(self, x, last_rotation_matrix=None):
        b, s, c = x.shape
        position_embedding = self.to_grid.grid_position_embedding.view(1, s, 3).to(device=x.device)
        
        if last_rotation_matrix is not None:
            position_embedding = torch.einsum('bci, bij-> bcj', position_embedding, last_rotation_matrix)
        
        position_embedding = position_embedding.expand(b, -1, -1)
        return torch.cat([x, position_embedding], dim=-1)


def CalcSpherePoints(num_points: int, device: str = "cpu") -> torch.Tensor:
    goldenRatio = (1 + 5**0.5) / 2
    i = torch.arange(num_points, device=device).view(-1, 1)
    theta = 2 * math.pi * i / goldenRatio
    phi = torch.arccos(1 - 2 * (i + 0.5) / num_points)
    points = torch.cat(
        [
            torch.cos(theta) * torch.sin(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(phi),
        ],
        dim=1,
    )
    # weight the points by their density
    pt_cross = points.view(1, -1, 3) - points.view(-1, 1, 3)
    pt_cross = torch.sum(pt_cross**2, dim=2)
    pt_cross = torch.exp(-pt_cross / (0.5 * 0.3))
    scalar = 1.0 / torch.sum(pt_cross, dim=1)
    scalar = num_points * scalar / torch.sum(scalar)
    res = points * (scalar.view(-1, 1))
    return res, -res

def CalcSpherePointsV2(n_points):
    """
        Calculating Fibonacci Lattices sphere V2
    """
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians
    
    for i in range(n_points):
        y = 1 - (i / float(n_points - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y
        
        theta = phi * i  # golden angle increment
        
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        
        points.append([x, y, z])
    
    return np.array(points)

def compute_repulsion_forces(points, k=1.0):
    """Optimize spherical points using force"""
    n = len(points)
    forces = np.zeros_like(points)
    
    for i in range(n):
        force = np.zeros(3)
        for j in range(n):
            if i != j:
                diff = points[i] - points[j]
                dist = np.linalg.norm(diff)
                if dist > 1e-10:  
                    force += k * diff / (dist ** 3)
        forces[i] = force
    
    return forces

def project_to_sphere(points):
    """Project to sphere"""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / norms

def sphere_uniformization_repulsion(points, iterations=1000, learning_rate=0.01, k=1.0):
    """Optimize spherical points"""
    
    for iteration in range(iterations):
        forces = compute_repulsion_forces(points, k)
        points += learning_rate * forces
        points = project_to_sphere(points)
        if iteration % 100 == 0:
            print(f"Optimizing spherical points: {iteration}/{iterations}")
    
    return points

class ToS2Grid_block:
    def __init__(self, l, res, device='cuda', EMPP=False, optim_step=500):
        self.grid_res = res
        self.device = device
        self.sampling_num = res
 
        # sphere_points, neg_sphere_points = CalcSpherePoints(self.sampling_num, 'cuda')
        
        sphere_points = CalcSpherePointsV2(self.sampling_num)
        if optim_step > 0:
            sphere_points = sphere_uniformization_repulsion(sphere_points, iterations=optim_step, learning_rate=0.01, k=0.01)
        sphere_points = torch.tensor(sphere_points, dtype=torch.float32, device=self.device)
        
        self.l = l
        sphharm_weights = o3.spherical_harmonics(
            torch.arange(0, l + 1).tolist(), sphere_points, normalize=True
        ).detach().unsqueeze(0)

        int_sphere = (sphharm_weights.view(self.sampling_num, -1) * sphharm_weights.view(self.sampling_num, -1)).sum(0, keepdim=True)
        int_sphere = torch.sqrt(int_sphere)

        # to grid
        if EMPP:
            to_grid_mat = sphharm_weights.view(self.sampling_num, -1)
        else:
            to_grid_mat = sphharm_weights.view(self.sampling_num, -1) / int_sphere
        self.to_grid_mat = nn.Parameter(data=to_grid_mat, requires_grad=False)

        from_grid_mat = to_grid_mat
        self.from_grid_mat = nn.Parameter(data=from_grid_mat, requires_grad=False)

        self.grid_position_embedding = sphere_points.view(1, self.sampling_num, 1, 3).contiguous()

    def ToGrid(self, x, last_rotation_matrix=None):
        to_grid_mat = self.to_grid_mat.to(device=x.device)
        
        if last_rotation_matrix is not None:
            wigner_d_matrix = RotationToWignerDMatrix(last_rotation_matrix, 0, self.l)
            to_grid_mat = torch.einsum('si,bij->bsj', to_grid_mat, wigner_d_matrix)
            return torch.einsum("zsi,zic->zsc", to_grid_mat, x).contiguous()
        
        return torch.einsum("si,zic->zsc", to_grid_mat, x).contiguous()
    
    def FromGrid(self, x_grid, last_rotation_matrix=None):
        from_grid_mat = self.from_grid_mat.to(device=x_grid.device)
        
        if last_rotation_matrix is not None:
            wigner_d_matrix = RotationToWignerDMatrix(last_rotation_matrix, 0, self.l)
            from_grid_mat = torch.einsum('si,bij->bsj', from_grid_mat, wigner_d_matrix)
            return torch.einsum("zsi,zsc->zic", from_grid_mat, x_grid).contiguous()
            
        return torch.einsum("si,zsc->zic", from_grid_mat, x_grid).contiguous()

def wigner_D(l, alpha, beta, gamma):
    if not l < len(_Jd):
        raise NotImplementedError(
            f"wigner D maximum l implemented is {len(_Jd) - 1}, send us an email to ask for more"
        )

    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    J = _Jd[l].to(dtype=alpha.dtype, device=alpha.device)
    Xa = _z_rot_mat(alpha, l)
    Xb = _z_rot_mat(beta, l)
    Xc = _z_rot_mat(gamma, l)
    return Xa @ J @ Xb @ J @ Xc

def _z_rot_mat(angle, l):
    shape, device, dtype = angle.shape, angle.device, angle.dtype
    M = angle.new_zeros((*shape, 2 * l + 1, 2 * l + 1))
    inds = torch.arange(0, 2 * l + 1, 1, device=device)
    reversed_inds = torch.arange(2 * l, -1, -1, device=device)
    frequencies = torch.arange(l, -l - 1, -1, dtype=dtype, device=device)
    M[..., inds, reversed_inds] = torch.sin(frequencies * angle[..., None])
    M[..., inds, inds] = torch.cos(frequencies * angle[..., None])
    return M
    
def RotationToWignerDMatrix(edge_rot_mat, start_lmax, end_lmax):
    x = edge_rot_mat @ edge_rot_mat.new_tensor([0.0, 1.0, 0.0])
    alpha, beta = o3.xyz_to_angles(x)
    R = (
        o3.angles_to_matrix(alpha, beta, torch.zeros_like(alpha)).transpose(-1, -2)
        @ edge_rot_mat
    )
    gamma = torch.atan2(R[..., 0, 2], R[..., 0, 0])

    size = (end_lmax + 1) ** 2 - (start_lmax) ** 2
    wigner = torch.zeros(len(alpha), size, size, device=x.device)
    start = 0
    for lmax in range(start_lmax, end_lmax + 1):
        block = wigner_D(lmax, alpha, beta, gamma)
        end = start + block.size()[1]
        wigner[:, start:end, start:end] = block
        start = end

    return wigner.detach()

    
def random_rotate(batch, device='cuda'):
    b = batch[-1] + 1
    r = generate_random_rotation_matrices_quaternion(batch_size=b, device=device)
    r = r[batch]
    return r

def generate_random_rotation_matrices_quaternion(batch_size, device):

    quaternions = torch.randn(batch_size, 4, device=device)
    
    quaternions = quaternions / torch.linalg.norm(quaternions, dim=1, keepdim=True)
    
    w = quaternions[:, 0]
    x = quaternions[:, 1]
    y = quaternions[:, 2]
    z = quaternions[:, 3]

    # w2, x2, y2, z2 = w * w, x * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    
    # R_00 = 1 - 2y^2 - 2z^2
    R00 = 1 - 2 * (y*y + z*z)
    R01 = 2 * (xy - wz)
    R02 = 2 * (xz + wy)
    
    # R_10 = 2x^y + 2wz
    R10 = 2 * (xy + wz)
    R11 = 1 - 2 * (x*x + z*z) # R_11 = 1 - 2x^2 - 2z^2
    R12 = 2 * (yz - wx)
    
    # R_20 = 2xz - 2wy
    R20 = 2 * (xz - wy)
    R21 = 2 * (yz + wx)
    R22 = 1 - 2 * (x*x + y*y) # R_22 = 1 - 2x^2 - 2y^2
    
    R = torch.stack((R00, R01, R02,
                     R10, R11, R12,
                     R20, R21, R22), dim=1)
    
    return R.view(batch_size, 3, 3)

class pos_prediction(torch.nn.Module):
    def __init__(self, 
                 sphere_channels, 
                 ffn_hidden_channels, 
                 out_channel,
                 lmax_list,
                 mmax_list,
                 SO3_grid,
                 ffn_activation,
                 use_gate_act,
                 use_grid_mlp,
                 use_sep_s2_act,
                 max_num_elements, 
                 num_gaussians=600, 
                 res=80
                 ):
        super().__init__()
        self.sphere_channels = sphere_channels
        self.ffn_hidden_channels = ffn_hidden_channels
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.SO3_grid = SO3_grid
        self.ffn_activation = ffn_activation
        self.use_gate_act = use_gate_act
        self.use_grid_mlp = use_grid_mlp
        self.use_sep_s2_act = use_sep_s2_act

        self.energy_gaussian = GaussianSmearing(num_gaussians=num_gaussians)
        self.energy_embedding = nn.Linear(num_gaussians, sphere_channels)
        self.atom_embedding = nn.Embedding(max_num_elements, sphere_channels)
        self.act = nn.SiLU()
        self.fc_embedding = nn.Linear(sphere_channels, sphere_channels)

        self.temperature = 0.1
        self.temperature_label = 0.1
        self.act = nn.SiLU()

        self.out_channel = out_channel
        self.ffn = FeedForwardNetwork(self.sphere_channels,
            self.ffn_hidden_channels,
            self.out_channel,
            self.lmax_list,
            self.mmax_list,
            self.SO3_grid,
            self.ffn_activation,
            self.use_gate_act,
            self.use_grid_mlp,
            self.use_sep_s2_act,
            )

        self.s2 = ToS2Grid_block(max(self.lmax_list), 512, EMPP=True, optim_step=100)
        self.fc_logit = nn.Sequential(nn.Linear(self.out_channel, out_channel // 4),
            nn.SiLU(),
            nn.Linear(out_channel // 4, 1))
        self.KLDivLoss = torch.nn.KLDivLoss()

    def forward(self, node_features, pos, mask_position, edge_index_mask, batch, target):
        energy, mask_atom = target
        energy_embedding = self.energy_gaussian(energy).to(dtype=pos.dtype)
        energy_embedding = self.energy_embedding(energy_embedding)
        mask_atom_embedding = self.atom_embedding(mask_atom)
        target_embedding = self.fc_embedding(self.act((energy_embedding + mask_atom_embedding)))
        node_features.embedding[:, 0, :] = node_features.embedding[:, 0, :] + target_embedding[batch]

        node_features.embedding = torch.index_select(node_features.embedding, 0, edge_index_mask[0])
        node_features = self.ffn(node_features)
        position_logit = self.s2.ToGrid(node_features.embedding) # [batch, b, a, c]
        position_logit = position_logit.reshape(position_logit.shape[0], -1, self.out_channel)
        position_logit = self.fc_logit(position_logit)
        position_logit = position_logit.squeeze()
        res = F.log_softmax(position_logit / self.temperature, 1)

        mask_position = mask_position[edge_index_mask[1]]
        neighbor_pos = pos[edge_index_mask[0]]
        label_pos = mask_position - neighbor_pos
        label_pos = label_pos.detach()

        label_pos = o3.spherical_harmonics(torch.arange(0, max(self.lmax_list) + 1).tolist(), label_pos, False)
        label_logit = self.s2.ToGrid(label_pos.unsqueeze(-1))
        label_logit = label_logit.reshape(label_logit.shape[0], -1)
        label_logit = F.softmax(label_logit / self.temperature_label, -1)
        return self.KLDivLoss(res, label_logit)

class GaussianSmearing(torch.nn.Module):
    def __init__(
        self,
        start: float = -10.0,
        stop: float = 10.0,
        num_gaussians: int = 50,
        basis_width_scalar: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_output = num_gaussians
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (basis_width_scalar * (offset[1] - offset[0])).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist) -> torch.Tensor:

        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))