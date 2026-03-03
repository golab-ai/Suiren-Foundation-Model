from __future__ import annotations

import contextlib
import math

import torch
import torch.nn as nn
from torch_cluster import radius

with contextlib.suppress(ImportError):
    pass

from .base import GraphModelMixin 
from .edge_rot_mat import init_edge_rot_mat
from .gaussian_rbf import GaussianRadialBasisLayer
from .input_block import EdgeDegreeEmbedding
from .layer_norm import (
    EquivariantLayerNormArray,
    EquivariantLayerNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonicsV2,
    get_normalization_layer,
)
from .module_list import ModuleListInfo
from .radial_function import RadialFunction
from .so3 import (
    CoefficientMappingModule,
    SO3_Embedding,
    SO3_Grid,
    SO3_LinearV2,
    SO3_Rotation,
)
from .transformer_block import (
    FeedForwardNetwork,
    SO2EquivariantGraphAttention,
    TransBlockV2,
    ToS2Grid_block,
    pos_prediction,
)

_AVG_NUM_NODES = 35.2160
_AVG_DEGREE = 26.44  

class GaussianSmearing(torch.nn.Module):
    def __init__(
        self,
        start: float = -5.0,
        stop: float = 5.0,
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

class EST_Eqv2(nn.Module, GraphModelMixin):
    """
        The model class of Equivariant Spherical Transformer (EST) and Equiformerv2.
    """
    def __init__(
        self,
        # hypermeters for Equiformer part
        use_pbc: bool = True,
        use_pbc_single: bool = False,
        regress_forces: bool = True,
        otf_graph: bool = True,
        max_neighbors: int = 500,
        max_radius: float = 5.0,
        max_num_elements: int = 90,
        num_layers: int = 12,
        sphere_channels: int = 128,
        attn_hidden_channels: int = 128,
        num_heads: int = 8,
        attn_alpha_channels: int = 32,
        attn_value_channels: int = 16,
        ffn_hidden_channels: int = 512,
        norm_type: str = "rms_norm_sh",
        lmax_list: list[int] | None = None,
        mmax_list: list[int] | None = None,
        grid_resolution: int | None = None,
        EST_grid_resolution: int | None = None,
        EST_grid_optim_step: int = 0,
        num_sphere_samples: int = 128,
        edge_channels: int = 128,
        use_atom_edge_embedding: bool = True,
        share_atom_edge_embedding: bool = False,
        use_m_share_rad: bool = False,
        distance_function: str = "gaussian",
        num_distance_basis: int = 512,
        attn_activation: str = "scaled_silu",
        use_s2_act_attn: bool = False,
        use_attn_renorm: bool = True,
        ffn_activation: str = "scaled_silu",
        use_gate_act: bool = False,
        use_grid_mlp: bool = False,
        use_sep_s2_act: bool = True,
        alpha_drop: float = 0.1,
        drop_path_rate: float = 0.05,
        proj_drop: float = 0.0,
        weight_init: str = "normal",
        enforce_max_neighbors_strictly: bool = True,
        avg_num_nodes: float | None = None,
        avg_degree: float | None = None,
        use_energy_lin_ref: bool | None = False,
        load_energy_lin_ref: bool | None = False,
        # hypermeters for EST part
        num_experts_steerable: int = 1,
        num_experts_spherical: int = 1,
        # others
        checkpoint: bool = False,
        EMPP: bool = False,
        output_module: bool = True,
    ):
        if mmax_list is None:
            mmax_list = [2]
        if lmax_list is None:
            lmax_list = [6]
        super().__init__()

        self.num_experts_steerable = num_experts_steerable
        self.num_experts_spherical = num_experts_spherical
        self.use_pbc = use_pbc
        self.use_pbc_single = use_pbc_single
        self.regress_forces = regress_forces
        self.otf_graph = otf_graph
        self.max_neighbors = max_neighbors
        self.max_radius = max_radius
        self.cutoff = max_radius
        self.max_num_elements = max_num_elements

        self.num_layers = num_layers
        self.sphere_channels = sphere_channels
        self.attn_hidden_channels = attn_hidden_channels
        self.num_heads = num_heads
        self.attn_alpha_channels = attn_alpha_channels
        self.attn_value_channels = attn_value_channels
        self.ffn_hidden_channels = ffn_hidden_channels
        self.norm_type = norm_type

        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.grid_resolution = grid_resolution

        self.num_sphere_samples = num_sphere_samples

        self.edge_channels = edge_channels
        self.use_atom_edge_embedding = use_atom_edge_embedding
        self.share_atom_edge_embedding = share_atom_edge_embedding
        if self.share_atom_edge_embedding:
            assert self.use_atom_edge_embedding
            self.block_use_atom_edge_embedding = False
        else:
            self.block_use_atom_edge_embedding = self.use_atom_edge_embedding
        self.use_m_share_rad = use_m_share_rad
        self.distance_function = distance_function
        self.num_distance_basis = num_distance_basis

        self.attn_activation = attn_activation
        self.use_s2_act_attn = use_s2_act_attn
        self.use_attn_renorm = use_attn_renorm
        self.ffn_activation = ffn_activation
        self.use_gate_act = use_gate_act
        self.use_grid_mlp = use_grid_mlp
        self.use_sep_s2_act = use_sep_s2_act

        self.alpha_drop = alpha_drop
        self.drop_path_rate = drop_path_rate
        self.proj_drop = proj_drop

        self.avg_num_nodes = avg_num_nodes or _AVG_NUM_NODES
        self.avg_degree = avg_degree or _AVG_DEGREE

        self.use_energy_lin_ref = use_energy_lin_ref
        self.load_energy_lin_ref = load_energy_lin_ref
        assert not (
            self.use_energy_lin_ref and not self.load_energy_lin_ref
        ), "You can't have use_energy_lin_ref = True and load_energy_lin_ref = False, since the model will not have the parameters for the linear references. All other combinations are fine."

        self.weight_init = weight_init
        assert self.weight_init in ["normal", "uniform"]

        self.enforce_max_neighbors_strictly = enforce_max_neighbors_strictly
        
        self.checkpoint = checkpoint

        self.device = torch.cuda.current_device()

        self.grad_forces = False
        self.num_resolutions: int = len(self.lmax_list)
        self.sphere_channels_all: int = self.num_resolutions * self.sphere_channels

        # Weights for message initialization
        self.sphere_embedding = nn.Embedding(
            self.max_num_elements, self.sphere_channels_all
        )

        # Initialize the function used to measure the distances between atoms
        assert self.distance_function in [
            "gaussian",
        ]
        if self.distance_function == "gaussian":
            self.distance_expansion = GaussianSmearing(
                0.0,
                self.cutoff,
                600,
                2.0,
            )
        else:
            raise ValueError

        # Initialize the sizes of radial functions (input channels and 2 hidden channels)
        self.edge_channels_list = [int(self.distance_expansion.num_output)] + [
            self.edge_channels
        ] * 2

        # Initialize atom edge embedding
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            self.source_embedding = nn.Embedding(
                self.max_num_elements, self.edge_channels_list[-1]
            )
            self.target_embedding = nn.Embedding(
                self.max_num_elements, self.edge_channels_list[-1]
            )
            self.edge_channels_list[0] = (
                self.edge_channels_list[0] + 2 * self.edge_channels_list[-1]
            )
        else:
            self.source_embedding, self.target_embedding = None, None

        # Initialize the module that compute WignerD matrices and other values for spherical harmonic calculations
        self.SO3_rotation_stand = nn.ModuleList()
        for i in range(self.num_resolutions):
            self.SO3_rotation_stand.append(SO3_Rotation(self.lmax_list[i]))

        # Initialize conversion between degree l and order m layouts
        self.mappingReduced = CoefficientMappingModule(self.lmax_list, self.mmax_list)

        # Initialize the transformations between spherical and grid representations
        self.SO3_grid_up = ModuleListInfo(
            f"({max(self.lmax_list)}, {max(self.lmax_list)})"
        )
        for lval in range(max(self.lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(max(self.lmax_list) + 1):
                SO3_m_grid.append(
                    SO3_Grid(
                        lval,
                        m,
                        resolution=self.grid_resolution,
                        normalization="component",
                    )
                )
            self.SO3_grid_up.append(SO3_m_grid)

        self.SO3_rotation_empp = nn.ModuleList()
        for i in range(self.num_resolutions):
            self.SO3_rotation_empp.append(SO3_Rotation(self.lmax_list[i]))
        
        self.SO3_rotation = [self.SO3_rotation_stand, self.SO3_rotation_empp]

        # Edge-degree embedding
        self.edge_degree_embedding = EdgeDegreeEmbedding(
            self.sphere_channels,
            self.lmax_list,
            self.mmax_list,
            self.SO3_rotation,
            self.mappingReduced,
            self.max_num_elements,
            self.edge_channels_list,
            self.block_use_atom_edge_embedding,
            rescale_factor=self.avg_degree,
        )

        # Initialize the grid distribution of EST
        if EST_grid_resolution:
            self.tos2grid = ToS2Grid_block(max(self.lmax_list), EST_grid_resolution, optim_step=EST_grid_optim_step)
        else:
            self.tos2grid = ToS2Grid_block(max(self.lmax_list), (2 * max(self.lmax_list) + 1) * (2 * max(self.lmax_list) + 2), optim_step=EST_grid_optim_step)
        # Initialize the blocks for each layer
        self.blocks = nn.ModuleList()
        self.mid_layers = int(1.0 * self.num_layers)
        for _ in range(self.num_layers):
            block = TransBlockV2(
                self.sphere_channels,
                self.attn_hidden_channels,
                self.num_heads,
                self.attn_alpha_channels,
                self.attn_value_channels,
                self.ffn_hidden_channels,
                self.sphere_channels,
                self.lmax_list,
                self.mmax_list,
                self.SO3_rotation,
                self.mappingReduced,
                self.SO3_grid_up,
                self.max_num_elements,
                self.edge_channels_list,
                self.block_use_atom_edge_embedding,
                self.use_m_share_rad,
                self.attn_activation,
                self.use_s2_act_attn,
                self.use_attn_renorm,
                self.ffn_activation,
                self.use_gate_act,
                self.use_grid_mlp,
                self.use_sep_s2_act,
                self.norm_type,
                self.alpha_drop,
                self.drop_path_rate,
                self.proj_drop,
                self.num_experts_steerable,
                self.num_experts_spherical,
                self.tos2grid,
            )
            self.blocks.append(block)

        # Output blocks for energy and forces
        self.norm = get_normalization_layer(
            self.norm_type,
            lmax=max(self.lmax_list),
            num_channels=self.sphere_channels,
        )
        
        self.output_module = output_module
        self.energy_block = FeedForwardNetwork(
            self.sphere_channels,
            self.ffn_hidden_channels,
            1,
            self.lmax_list,
            self.mmax_list,
            self.SO3_grid_up,
            self.ffn_activation,
            self.use_gate_act,
            self.use_grid_mlp,
            self.use_sep_s2_act,
            norm_flag=False,
        )
        if self.regress_forces:
            self.force_block = SO2EquivariantGraphAttention(
                self.sphere_channels,
                self.attn_hidden_channels,
                self.num_heads,
                self.attn_alpha_channels,
                self.attn_value_channels,
                1,
                self.lmax_list,
                self.mmax_list,
                self.SO3_rotation,
                self.mappingReduced,
                self.SO3_grid_up,
                self.max_num_elements,
                self.edge_channels_list,
                self.block_use_atom_edge_embedding,
                self.use_m_share_rad,
                self.attn_activation,
                self.use_s2_act_attn,
                self.use_attn_renorm,
                self.use_gate_act,
                self.use_sep_s2_act,
                alpha_drop=0.0,
            )       

        if EMPP:
            self.position_prediction = pos_prediction(
                    self.sphere_channels,
                    self.ffn_hidden_channels,
                    64,
                    self.lmax_list,
                    self.mmax_list,
                    self.SO3_grid_up,
                    self.ffn_activation,
                    self.use_gate_act,
                    self.use_grid_mlp,
                    self.use_sep_s2_act,
                    self.max_num_elements,
                    self.num_distance_basis,
                    )

        if self.load_energy_lin_ref:
            self.energy_lin_ref = nn.Parameter(
                torch.zeros(self.max_num_elements),
                requires_grad=False,
            )

        self.apply(self._init_weights)
        self.apply(self._uniform_init_rad_func_linear_weights)

    def forward(self, data, auxiliary_target_name=None, normalizers=None):
        output = self.standard_forward(data)
        if auxiliary_target_name is not None:
            new_data = self.add_mask(data, auxiliary_target_name, normalizers)
            auxiliary_loss = self.auxiliary_forward(new_data)
            return output, auxiliary_loss
        return output

    def standard_forward(self, data):
        data.natoms = data.ptr[1:] - data.ptr[:-1]
        self.batch_size = len(data.natoms)
        data.pos = data.pos.to(dtype=torch.float32)
        self.dtype = data.pos.dtype
        self.device = data.pos.device
        atomic_numbers = data.x.long()
        data.atomic_numbers = atomic_numbers

        graph = self.generate_graph(
            data,
            enforce_max_neighbors_strictly=self.enforce_max_neighbors_strictly,
        )


        data_batch = data.batch

        ###############################################################
        # Initialize data structures
        ###############################################################

        # Compute 3x3 rotation matrix per edge
        edge_rot_mat = self._init_edge_rot_mat(
            data, graph.edge_index, graph.edge_distance_vec
        )

        # Initialize the WignerD matrices and other values for spherical harmonic calculations
        for i in range(self.num_resolutions):
            self.SO3_rotation[0][i].set_wigner(edge_rot_mat)

        ###############################################################
        # Initialize node embeddings
        ###############################################################

        # Init per node representations using an atomic number based embedding
        x = SO3_Embedding(
            len(atomic_numbers),
            self.lmax_list,
            self.sphere_channels,
            self.device,
            self.dtype,
        )

        offset_res = 0
        offset = 0
        # Initialize the l = 0, m = 0 coefficients for each resolution
        for i in range(self.num_resolutions):
            if self.num_resolutions == 1:
                x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)
            else:
                x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)[
                    :, offset : offset + self.sphere_channels
                ]
            offset = offset + self.sphere_channels
            offset_res = offset_res + int((self.lmax_list[i] + 1) ** 2)

        # Edge encoding (distance and atom edge)
        graph.edge_distance = self.distance_expansion(graph.edge_distance)
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            source_element = graph.atomic_numbers_full[
                graph.edge_index[0]
            ]  # Source atom atomic number
            target_element = graph.atomic_numbers_full[
                graph.edge_index[1]
            ]  # Target atom atomic number
            source_embedding = self.source_embedding(source_element)
            target_embedding = self.target_embedding(target_element)
            graph.edge_distance = torch.cat(
                (graph.edge_distance, source_embedding, target_embedding), dim=1
            )

        # Edge-degree embedding
        edge_degree = self.edge_degree_embedding(
            graph.atomic_numbers_full,
            graph.edge_distance,
            graph.edge_index,
            len(atomic_numbers),
            graph.node_offset,
            ssp=False,
        )
        x.embedding = x.embedding + edge_degree.embedding

        ###############################################################
        # Update spherical node embeddings
        ###############################################################

        for i in range(self.num_layers):
            if not self.checkpoint:
                x = self.blocks[i](
                    x,  # SO3_Embedding
                    graph.atomic_numbers_full,
                    graph.edge_distance,
                    graph.edge_index,
                    batch=data_batch,  # for GraphDropPath
                    node_offset=graph.node_offset,
                    ssp=False,
                )
            else:
                x = torch.utils.checkpoint.checkpoint(
                        self.blocks[i],
                        x,  # SO3_Embedding
                        graph.atomic_numbers_full,
                        graph.edge_distance,
                        graph.edge_index,
                        data_batch,  # for GraphDropPath
                        graph.node_offset,
                        False,
                        use_reentrant=not self.training,
                    )

        # Final layer norm
        x.embedding = self.norm(x.embedding)

        if self.output_module:
            ###############################################################
            # Energy estimation
            ###############################################################
            node_energy = self.energy_block(x)
            node_energy = node_energy.embedding.narrow(1, 0, 1)
            energy = torch.zeros(
                len(data.natoms),
                device=node_energy.device,
                dtype=node_energy.dtype,
            )
            energy.index_add_(0, graph.batch_full, node_energy.view(-1))
            energy = energy / self.avg_num_nodes

            # Add the per-atom linear references to the energy.
            if self.use_energy_lin_ref and self.load_energy_lin_ref:
                # During training, target E = (E_DFT - E_ref - E_mean) / E_std, and
                # during inference, \hat{E_DFT} = \hat{E} * E_std + E_ref + E_mean
                # where
                #
                # E_DFT = raw DFT energy,
                # E_ref = reference energy,
                # E_mean = normalizer mean,
                # E_std = normalizer std,
                # \hat{E} = predicted energy,
                # \hat{E_DFT} = predicted DFT energy.
                #
                # We can also write this as
                # \hat{E_DFT} = E_std * (\hat{E} + E_ref / E_std) + E_mean,
                # which is why we save E_ref / E_std as the linear reference.
                with torch.autocast("cuda", enabled=False):
                    energy = energy.to(self.energy_lin_ref.dtype).index_add(
                        0,
                        graph.batch_full,
                        self.energy_lin_ref[graph.atomic_numbers_full],
                    )

            outputs = {"energy": energy}

            ###############################################################
            # Force estimation
            ###############################################################
            forces = self.force_block(
                x,
                graph.atomic_numbers_full,
                graph.edge_distance,
                graph.edge_index,
                node_offset=graph.node_offset,
                ssp=False,
            )
            forces = forces.embedding.narrow(1, 1, 3)
            forces = forces.view(-1, 3).contiguous()

            outputs["forces"] = forces
            outputs["node_embedding"] = x.embedding
            return outputs
        else:
            return {"node_embedding": x.embedding}

    def add_mask(self, data, target_name='energy', normalizers=None):
        # EMPP mask (one masked atom per molecule)
        mask = torch.tensor([], dtype=torch.long).to(device=data.ptr.device)
        for i in range(len(data.ptr) - 1):
            length = data.ptr[i + 1] - data.ptr[i]
            value = torch.randperm(length).to(device=data.ptr.device) + data.ptr[i]
            mask = torch.cat([mask, value[0:1]])
        mask_inverse = torch.tensor([i for i in range(data.pos.size(0)) if i not in mask], device=data.pos.device)

        new_data = data.clone()
        new_data.mask_pos = data.pos[mask]
        new_data.mask_pos_atom = data.x[mask]
        new_data.mask = mask
        new_data.pos = torch.index_select(data.pos, 0, mask_inverse)
        new_data.ori_x = data.x.detach()
        new_data.x = torch.index_select(data.x, 0, mask_inverse)
        new_data.ori_batch = data.batch
        new_data.batch = torch.index_select(data.batch, 0, mask_inverse)
        target = data[target_name]

        target = normalizers[target_name].norm(target)
        new_data.target = (target, new_data.mask_pos_atom)
        new_data.ptr[1:] = data.ptr[1:] - 1

        return new_data

    def auxiliary_forward(self, data):
        data.natoms = data.ptr[1:] - data.ptr[:-1]
        self.batch_size = len(data.natoms)
        data.pos = data.pos.to(dtype=torch.float32)
        self.dtype = data.pos.dtype
        self.device = data.pos.device
        atomic_numbers = data.x.long()
        data.atomic_numbers = atomic_numbers

        graph = self.generate_graph(
            data,
            enforce_max_neighbors_strictly=self.enforce_max_neighbors_strictly,
        )

        data_batch = data.batch

        ###############################################################
        # Initialize data structures
        ###############################################################

        # Compute 3x3 rotation matrix per edge
        edge_rot_mat = self._init_edge_rot_mat(
            data, graph.edge_index, graph.edge_distance_vec
        )

        # Initialize the WignerD matrices and other values for spherical harmonic calculations
        for i in range(self.num_resolutions):
            self.SO3_rotation[1][i].set_wigner(edge_rot_mat)

        ###############################################################
        # Initialize node embeddings
        ###############################################################

        # Init per node representations using an atomic number based embedding
        x = SO3_Embedding(
            len(atomic_numbers),
            self.lmax_list,
            self.sphere_channels,
            self.device,
            self.dtype,
        )

        # batch_x = torch.arange(self.batch_size + 1, device=data.pos.device)
        # batch_y = torch.bucketize(batch_x, data.batch)
        # edge_mask_index = torch.ops.torch_cluster.radius(data.mask_pos, data.pos, batch_x, batch_y, self.max_radius, 50, 1)
        batch_x = torch.arange(self.batch_size, device=data.pos.device)
        edge_mask_index = radius(data.mask_pos, data.pos, self.max_radius, batch_x, data.batch, max_num_neighbors=50)
        target = data.target

        offset_res = 0
        offset = 0
        # Initialize the l = 0, m = 0 coefficients for each resolution
        for i in range(self.num_resolutions):
            if self.num_resolutions == 1:
                x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)
            else:
                x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)[
                    :, offset : offset + self.sphere_channels
                ]
            offset = offset + self.sphere_channels
            offset_res = offset_res + int((self.lmax_list[i] + 1) ** 2)

        # Edge encoding (distance and atom edge)
        graph.edge_distance = self.distance_expansion(graph.edge_distance)
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            source_element = graph.atomic_numbers_full[
                graph.edge_index[0]
            ]  # Source atom atomic number
            target_element = graph.atomic_numbers_full[
                graph.edge_index[1]
            ]  # Target atom atomic number
            source_embedding = self.source_embedding(source_element)
            target_embedding = self.target_embedding(target_element)
            graph.edge_distance = torch.cat(
                (graph.edge_distance, source_embedding, target_embedding), dim=1
            )

        # Edge-degree embedding
        edge_degree = self.edge_degree_embedding(
            graph.atomic_numbers_full,
            graph.edge_distance,
            graph.edge_index,
            len(atomic_numbers),
            graph.node_offset,
            ssp=True,
        )
        x.embedding = x.embedding + edge_degree.embedding

        ###############################################################
        # Update spherical node embeddings
        ###############################################################

        for i in range(self.num_layers):
            if i > self.mid_layers:
                x = self.blocks[i](
                    x,  # SO3_Embedding
                    graph.atomic_numbers_full,
                    graph.edge_distance,
                    graph.edge_index,
                    batch=data_batch,  # for GraphDropPath
                    node_offset=graph.node_offset,
                    ssp=True,
                )
            else:
                x = torch.utils.checkpoint.checkpoint(
                        self.blocks[i],
                        x,  # SO3_Embedding
                        graph.atomic_numbers_full,
                        graph.edge_distance,
                        graph.edge_index,
                        data_batch,  # for GraphDropPath
                        graph.node_offset,
                        True,
                        use_reentrant=not self.training,
                    )

        # Final layer norm
        x.embedding = self.norm(x.embedding)

        EMPP_loss = self.position_prediction(x, data.pos, data.mask_pos, edge_mask_index, data.batch, target)

        return EMPP_loss
        

    # Initialize the edge rotation matrics
    def _init_edge_rot_mat(self, data, edge_index, edge_distance_vec):
        return init_edge_rot_mat(edge_distance_vec)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, m):
        if isinstance(m, (torch.nn.Linear, SO3_LinearV2)):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            if self.weight_init == "normal":
                std = 1 / math.sqrt(m.in_features)
                torch.nn.init.normal_(m.weight, 0, std)

        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    def _uniform_init_rad_func_linear_weights(self, m):
        if isinstance(m, RadialFunction):
            m.apply(self._uniform_init_linear_weights)

    def _uniform_init_linear_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            std = 1 / math.sqrt(m.in_features)
            torch.nn.init.uniform_(m.weight, -std, std)

    @torch.jit.ignore
    def no_weight_decay(self) -> set:
        no_wd_list = []
        named_parameters_list = [name for name, _ in self.named_parameters()]
        for module_name, module in self.named_modules():
            if isinstance(
                module,
                (
                    torch.nn.Linear,
                    SO3_LinearV2,
                    torch.nn.LayerNorm,
                    EquivariantLayerNormArray,
                    EquivariantLayerNormArraySphericalHarmonics,
                    EquivariantRMSNormArraySphericalHarmonics,
                    EquivariantRMSNormArraySphericalHarmonicsV2,
                    GaussianRadialBasisLayer,
                ),
            ):
                for parameter_name, _ in module.named_parameters():
                    if (
                        isinstance(module, (torch.nn.Linear, SO3_LinearV2))
                        and "weight" in parameter_name
                    ):
                        continue
                    global_parameter_name = module_name + "." + parameter_name
                    assert global_parameter_name in named_parameters_list
                    no_wd_list.append(global_parameter_name)

        return set(no_wd_list)