import math
import os
import sys

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.nn import (
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.utils import scatter

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from data.utils import NUM_ATOM_TYPES, axis_angle_to_matrix


class COSMol_interaction(nn.Module):
    def __init__(self, input_dim, cfg):
        super().__init__()
        self.input_dim = input_dim
        self.dropout = cfg.dropout
        self.normalization = cfg.normalization
        self.without_vec_attn = cfg.get("without_vec_attn", False)

        # linear projections
        self.lin_Q = nn.Linear(input_dim, input_dim // 2)
        self.lin_K = nn.Linear(input_dim, input_dim // 2)
        self.lin_V = nn.Linear(input_dim, input_dim)

        if not self.without_vec_attn:
            self.lin_Q_vec = nn.Linear(input_dim, input_dim // 2, bias=False)
            self.lin_K_vec = nn.Linear(input_dim, input_dim // 2, bias=False)
        self.lin_V_vec = nn.Linear(input_dim, input_dim, bias=False)

        self.edge_V_j = nn.Linear(input_dim, input_dim)

        self.edge = nn.Linear(input_dim, input_dim)

        # edge feature encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim), nn.SiLU()
        )
        # compute per-edge filter weights
        self.filter_mlp = nn.Linear(input_dim, input_dim * 2)

        # directional coefficients
        self.direction_mlp = nn.Linear(input_dim, input_dim)

        if self.normalization != False:
            self.dyt_scalar = DyT(num_features=input_dim, use_bias=True)
            self.dyt_vector = VectorDyT(num_features=input_dim)

    def forward(
        self,
        x,  # (N, D)
        x_vec,  # (N, 3, D)
        edge_feat,  # (E, D)
        fcut,  # (E, 1) or (E,)
        edge_vector_normalized,  # (E, 3)
        edge_index,  # (2, E) src, dst
    ):
        # Optional DYT preprocessing
        x_, x_vec_ = (
            (self.dyt_scalar(x), self.dyt_vector(x_vec))
            if self.normalization == "preDyt"
            else (x, x_vec)
        )  # x_: (num_nodes, input_dim), x_vec_: (num_nodes, 3, input_dim)

        src, dst = edge_index[0], edge_index[1]
        N = x.size(0)
        D = self.input_dim

        # Node projections
        Q = self.lin_Q(x_)  # (N, D)
        K = self.lin_K(x_)  # (N, D)
        V = self.lin_V(x_)  # (N, D)

        if not self.without_vec_attn:
            Q_vec = self.lin_Q_vec(x_vec_)  # (N, 3, D)
            K_vec = self.lin_K_vec(x_vec_)  # (N, 3, D)
        V_vec = self.lin_V_vec(x_vec_)  # (N, 3, D)

        # index-select for edges
        Q_i = Q[dst]  # (E, D) - query at destination (attn from src -> dst)
        K_j = K[src]  # (E, D)
        V_j = V[src] + self.edge_V_j(edge_feat)  # (E, D)

        if not self.without_vec_attn:
            Qv_i = Q_vec[dst]  # (E, 3, D)
            Kv_j = K_vec[src] + edge_vector_normalized.unsqueeze(-1)  # (E, 3, D)
        Vv_j = V_vec[src]  # (E, 3, D)

        # basic elementwise interactions
        att_scalar = Q_i * K_j  # (E, D)
        if not self.without_vec_attn:
            att_vector = (Qv_i * Kv_j).sum(dim=1)  # (E, D) sum over 3 -> (E, D)

        # processed edge info
        if self.without_vec_attn:
            components = [att_scalar, att_scalar]
        else:
            components = [att_scalar, att_vector]

        att_input = torch.cat(components, dim=-1) * self.edge(edge_feat)  # (E, D_total)

        # MLP -> produce per-edge per-feature filter
        h_edge = self.edge_encoder(att_input)
        att_filter = self.filter_mlp(h_edge)  # (E, D)
        att_filter = F.dropout(
            att_filter, p=self.dropout, training=self.training
        ) * fcut.view(-1, 1)

        # Split filter into scalar and vector components
        att_filter_sca, att_filter_vec = torch.split(att_filter, D, dim=-1)

        # Apply filter to values and aggregate
        out_x = att_filter_sca * V_j  # (E, D)
        out_x = scatter(out_x, dst, dim=0, dim_size=N, reduce="sum")  # (N, D)

        # Vector branch: expand att_filter to align with (E, 3, D)
        out_x_vec = att_filter_vec.unsqueeze(1) * Vv_j  # (E, 3, D)
        out_x_vec = scatter(
            out_x_vec, dst, dim=0, dim_size=N, reduce="sum"
        )  # (N, 3, D)

        # edge directional contribution
        alpha_edge = F.silu(self.direction_mlp(h_edge)) * fcut.view(-1, 1)  # (E, D)
        d_edge_vec = alpha_edge.unsqueeze(1) * edge_vector_normalized.unsqueeze(
            -1
        )  # (E,3,D)
        d_edge_vec = scatter(
            d_edge_vec, dst, dim=0, dim_size=N, reduce="sum"
        )  # (N,3,D)

        # residuals
        out_x = out_x + x
        out_x_vec = out_x_vec + d_edge_vec + x_vec

        if self.normalization == "postDyt":
            return self.dyt_scalar(out_x), self.dyt_vector(out_x_vec)

        return out_x, out_x_vec


class CosineCutoff(nn.Module):
    def __init__(self, cutoff):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, distances):
        cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff) + 1.0)
        # remove contributions beyond the cutoff radius
        cutoffs = cutoffs * (distances < self.cutoff)
        return cutoffs


class CosineCutoff_2(nn.Module):
    def __init__(self, cutoff):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, distances):
        cutoffs = torch.cos(distances * math.pi / self.cutoff / 2)
        # remove contributions beyond the cutoff radius
        cutoffs = cutoffs * (distances < self.cutoff)
        return cutoffs


class GaussianSmearing(nn.Module):
    # From https://github.com/torchmd/torchmd-net/blob/fd329eed8d8125f31b9d776a9e8fd14861a58b04/torchmdnet/models/utils.py#L297
    def __init__(
        self,
        cutoff_lower=0.0,
        cutoff_upper=5.0,
        num_rbf=50,
        trainable=True,
        dtype=torch.float32,
    ):
        super(GaussianSmearing, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.trainable = trainable
        self.dtype = dtype
        offset, coeff = self._initial_params()
        if trainable:
            self.register_parameter("coeff", nn.Parameter(coeff))
            self.register_parameter("offset", nn.Parameter(offset))
        else:
            self.register_buffer("coeff", coeff)
            self.register_buffer("offset", offset)

    def _initial_params(self):
        offset = torch.linspace(
            self.cutoff_lower, self.cutoff_upper, self.num_rbf, dtype=self.dtype
        )
        coeff = -0.5 / torch.pow((offset[1] - offset[0]), 2)
        return offset, coeff

    def reset_parameters(self):
        offset, coeff = self._initial_params()
        self.offset.data.copy_(offset)
        self.coeff.data.copy_(coeff)

    def forward(self, dist):
        dist = dist - self.offset
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ExpNormalSmearing(nn.Module):
    def __init__(
        self,
        cutoff_lower=0.0,
        cutoff_upper=5.0,
        num_rbf=50,
        trainable=True,
        dtype=torch.float32,
    ):
        super(ExpNormalSmearing, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.trainable = trainable
        self.dtype = dtype
        self.cutoff_fn = CosineCutoff(0, cutoff_upper)
        self.alpha = 5.0 / (cutoff_upper - cutoff_lower)

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        # initialize means and betas according to the default values in PhysNet
        # https://pubs.acs.org/doi/10.1021/acs.jctc.9b00181
        start_value = torch.exp(
            torch.scalar_tensor(
                -self.cutoff_upper + self.cutoff_lower, dtype=self.dtype
            )
        )
        means = torch.linspace(start_value, 1, self.num_rbf, dtype=self.dtype)
        betas = torch.tensor(
            [(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf,
            dtype=self.dtype,
        )
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        dist = dist.unsqueeze(-1)
        return self.cutoff_fn(dist) * torch.exp(
            -self.betas
            * (torch.exp(self.alpha * (-dist + self.cutoff_lower)) - self.means) ** 2
        )


class Conformation_generation(torch.nn.Module):
    def __init__(self, input_dim, cfg):
        super().__init__()
        self.clamp_range = cfg.get("clamp_range", 1.0)
        self.input_dim = input_dim
        self.n_steps = cfg.get("n_steps", 0)
        self.output_layer = torch.nn.Linear(self.input_dim, 1)
        output_layer_input_dim = self.input_dim * 2 if self.n_steps else self.input_dim
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)
        self.weight = nn.Parameter(torch.ones(1) * self.clamp_range)

        layers = []
        for i in range(len(cfg.hidden_dim)):
            if i == 0:
                layers.append(
                    torch.nn.Linear(output_layer_input_dim, cfg.hidden_dim[i])
                )
            else:
                layers.append(torch.nn.Linear(cfg.hidden_dim[i - 1], cfg.hidden_dim[i]))
            if i != len(cfg.hidden_dim) - 1:
                layers.append(torch.nn.SiLU())
                layers.append(torch.nn.Dropout(cfg.dropout))
        self.output_mlp = torch.nn.Sequential(*layers)

        if self.n_steps:
            self.step_rbf = GaussianSmearing(
                0.0, self.n_steps, self.input_dim, trainable=True
            )

    def forward(self, input, pos, t):
        emb_scalar, emb_vector, batch = input
        coord_shift = self.output_layer(emb_vector).squeeze(-1)
        norm = torch.norm(coord_shift, dim=1, keepdim=True)
        scale_factor = 1 / (norm + 1e-5)

        if self.n_steps:
            emb_scalar = torch.cat(
                [emb_scalar, self.step_rbf((self.n_steps * t).unsqueeze(-1))], dim=-1
            )

        coord_shift = (
            coord_shift
            * scale_factor
            * torch.tanh(self.alpha * self.output_mlp(emb_scalar))
            * self.weight
        )

        return coord_shift


class Docking_output(torch.nn.Module):
    def __init__(self, input_dim, cfg):
        super().__init__()
        self.input_dim = input_dim
        self.n_steps = cfg.get("n_steps", 0)
        self.output_layer = torch.nn.Linear(self.input_dim, 1)
        output_layer_input_dim = self.input_dim * 2 if self.n_steps else self.input_dim
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)
        self.weight = nn.Parameter(torch.ones(1) * 5.0)

        layers = []
        for i in range(len(cfg.hidden_dim)):
            if i == 0:
                layers.append(
                    torch.nn.Linear(output_layer_input_dim, cfg.hidden_dim[i])
                )
            else:
                layers.append(torch.nn.Linear(cfg.hidden_dim[i - 1], cfg.hidden_dim[i]))
            if i != len(cfg.hidden_dim) - 1:
                layers.append(torch.nn.SiLU())
                layers.append(torch.nn.Dropout(cfg.dropout))
        self.output_mlp = torch.nn.Sequential(*layers)

        self.rotation_axis_0 = torch.nn.Linear(self.input_dim, self.input_dim)
        self.rotation_axis = torch.nn.Linear(self.input_dim, 1)
        self.shift = torch.nn.Linear(self.input_dim, self.input_dim)
        self.theta = torch.nn.Linear(output_layer_input_dim, 1)

        if self.n_steps:
            self.step_rbf = GaussianSmearing(
                0.0, self.n_steps, self.input_dim, trainable=True
            )

    def forward(self, input, pos, t, ligand_mask):
        emb_scalar, emb_vector, batch = input
        if batch is None:
            batch = torch.zeros(
                emb_scalar.size(0), dtype=torch.long, device=emb_scalar.device
            )

        ligand_mask = torch.nonzero(ligand_mask, as_tuple=False).view(-1)

        emb_scalar = emb_scalar[ligand_mask]
        emb_vector = emb_vector[ligand_mask]
        ligand_mask = ligand_mask

        coord_shift = self.output_layer(emb_vector).squeeze(-1)
        norm = torch.norm(coord_shift, dim=1, keepdim=True)
        scale_factor = 1 / (norm + 1e-5)

        if self.n_steps:
            emb_scalar = torch.cat(
                [emb_scalar, self.step_rbf((self.n_steps * t).unsqueeze(-1))], dim=-1
            )

        coord_shift = (
            coord_shift
            * scale_factor
            * torch.tanh(self.alpha * self.output_mlp(emb_scalar))
            * self.weight
        )

        # calculate rotation axis
        rotation_axis = self.rotation_axis(
            scatter(
                self.rotation_axis_0(emb_vector),
                batch[ligand_mask],
                dim=0,
                reduce="mean",
            )
        )

        # theta
        theta = self.theta(scatter(emb_scalar, batch[ligand_mask], dim=0, reduce="max"))

        R = axis_angle_to_matrix(rotation_axis, theta)

        shift_all = self.shift(
            scatter(emb_vector, batch[ligand_mask], dim=0, reduce="mean")
        ).mean(dim=-1)

        return coord_shift, R, shift_all


class Pocket_out(torch.nn.Module):
    def __init__(self, input_dim, cfg):
        super().__init__()
        self.output_layer = torch.nn.Linear(input_dim, 1)

        layers = []
        for i in range(len(cfg.hidden_dim)):
            if i == 0:
                layers.append(torch.nn.Linear(cfg.hidden_dim[0], cfg.hidden_dim[i]))
            else:
                layers.append(torch.nn.Linear(cfg.hidden_dim[i - 1], cfg.hidden_dim[i]))
            if i != len(cfg.hidden_dim) - 1:
                layers.append(torch.nn.SiLU())
                layers.append(torch.nn.Dropout(cfg.dropout))
        self.score = torch.nn.Sequential(*layers)

    def forward(self, input, CB_mask):
        emb_scalar, emb_vector, batch = input

        emb_scalar = emb_scalar[CB_mask]
        emb_vector = emb_vector[CB_mask]

        vec_to_pocket = self.output_layer(emb_vector).squeeze(-1)

        return vec_to_pocket


class Pocket_ranker_out(torch.nn.Module):
    def __init__(self, input_dim, cfg):
        super().__init__()

        layers = []
        for i in range(len(cfg.hidden_dim)):
            if i == 0:
                layers.append(torch.nn.Linear(input_dim, cfg.hidden_dim[i]))
            else:
                layers.append(torch.nn.Linear(cfg.hidden_dim[i - 1], cfg.hidden_dim[i]))
            if i != len(cfg.hidden_dim) - 1:
                layers.append(torch.nn.SiLU())
                layers.append(torch.nn.Dropout(cfg.dropout))
        self.score = torch.nn.Sequential(*layers)

    def forward(self, input, x):
        emb_scalar, emb_vector, batch = input

        # emb_scalar = emb_scalar
        # emb_vector = emb_vector
        emb_scalar = emb_scalar[x[:, 0] == NUM_ATOM_TYPES]

        score = self.score(emb_scalar)

        return score.squeeze(1)


class DyT(torch.nn.Module):
    def __init__(self, num_features, alpha_init_value=0.5, use_bias=True):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = torch.nn.Parameter(torch.ones(num_features) * 2)
        self.use_bias = use_bias
        if self.use_bias:
            self.bias = torch.nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        x = x * self.weight
        if self.use_bias:
            x = x + self.bias
        return x


class VectorDyT(nn.Module):
    def __init__(self, num_features, alpha_init_value=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features) * 2)

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, 3, num_features)
        Returns: Transformed tensor of the same shape
        """
        norm = torch.norm(x, dim=1, keepdim=True)
        scale_factor = torch.tanh(self.alpha * norm) / (norm + 1e-12)
        x = x * scale_factor * self.weight

        return x


class COSMol_update(torch.nn.Module):
    def __init__(self, input_dim, cfg):
        super().__init__()
        self.input_dim = input_dim
        self.use_dyt = cfg.get("use_dyt", False)
        self.wo_cross = cfg.get("wo_cross", False)
        self.channel_mix = torch.nn.Linear(input_dim, input_dim * 2, bias=False)
        self.linear_cross = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.intra_node_context_net = torch.nn.Sequential(
            torch.nn.Linear(input_dim * 2, input_dim * 2),
            torch.nn.SiLU(),
            torch.nn.Linear(
                input_dim * 2,
                input_dim * 2,
            ),
        )

        if self.use_dyt != False:
            self.dyt_scalar = DyT(num_features=input_dim, use_bias=True)
            self.dyt_vector = VectorDyT(num_features=input_dim)

    def forward(self, x, x_vec):
        if self.use_dyt == "preDyt":
            x_, x_vec_ = (self.dyt_scalar(x), self.dyt_vector(x_vec))
        else:
            x_, x_vec_ = (x, x_vec)

        x_vec_V, x_vec_W = torch.split(self.channel_mix(x_vec_), self.input_dim, dim=-1)
        mu_len = torch.norm(x_vec_V, dim=-2)

        comb = self.intra_node_context_net(torch.cat([x_, mu_len], dim=-1))

        d_vec_intra, d_x_vec_intra = torch.split(
            comb,
            self.input_dim,
            dim=-1,
        )

        d_vec_intra = x_vec_W * d_vec_intra.unsqueeze(-2)

        vec_cross = self.linear_cross(torch.cross(x_vec_V, x_vec_W, dim=-2))
        if self.wo_cross:
            vec_cross = torch.ones_like(vec_cross).to(x.device)

        x = x + d_x_vec_intra
        x_vec = x_vec + d_vec_intra + vec_cross

        if self.use_dyt == "postDyt":
            return self.dyt_scalar(x), self.dyt_vector(x_vec)

        return x, x_vec


class pose_ranker_out(torch.nn.Module):
    def __init__(self, input_dim, cfg):
        super().__init__()
        self.reduce = cfg.get("reduce", ["max"])
        input_dim = (input_dim) * len(self.reduce) * 2
        layers = []
        for i in range(len(cfg.hidden_dim)):
            if i == 0:
                layers.append(torch.nn.Linear(input_dim, cfg.hidden_dim[i]))
            else:
                layers.append(torch.nn.Linear(cfg.hidden_dim[i - 1], cfg.hidden_dim[i]))
            if i != len(cfg.hidden_dim) - 1:
                layers.append(torch.nn.SiLU())
                layers.append(torch.nn.Dropout(cfg.dropout))
        self.output_layer = torch.nn.Sequential(*layers)

    def forward(self, input, mol_mask):
        emb_scalar, emb_vector, batch = input
        emb_vector_ = emb_vector[mol_mask].norm(dim=-2)
        emb = torch.cat([emb_scalar[mol_mask], emb_vector_], dim=-1)
        concat = []
        if "max" in self.reduce:
            concat.append(global_max_pool(emb, batch[mol_mask]))
        if "mean" in self.reduce:
            concat.append(global_mean_pool(emb, batch[mol_mask]))
        if "sum" in self.reduce:
            concat.append(global_add_pool(emb, batch[mol_mask]))
        out = self.output_layer(torch.cat(concat, dim=-1))
        return out
