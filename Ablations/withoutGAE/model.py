import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (GCNConv, TAGConv, GATv2Conv, GINConv,
                                TransformerConv, global_add_pool)
from torch_geometric.utils import negative_sampling


def make_conv(conv_type, in_dim, out_dim):
    if conv_type == 'GCN':
        return GCNConv(in_dim, out_dim)
    elif conv_type == 'TAG':
        return TAGConv(in_dim, out_dim, K=2)
    elif conv_type == 'GATv2':
        return GATv2Conv(in_dim, out_dim, heads=1, dropout=0.2)
    elif conv_type == 'GIN':
        return GINConv(nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU()))
    elif conv_type == 'Transformer':
        return TransformerConv(in_dim, out_dim, heads=1, dropout=0.2)
    else:
        raise ValueError(f"Unknown conv_type: {conv_type}")


class ConvEncoder(nn.Module):
    def __init__(self, conv_type, layer_dims):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_dims) - 1):
            self.layers.append(make_conv(conv_type, layer_dims[i], layer_dims[i + 1]))

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = F.relu(layer(x, edge_index))
        return x


class PerPocketCrossAttnDecoder(nn.Module):
    def __init__(self, dim, num_heads=1):
        super().__init__()
        self.head_dim = dim
        self.q_proj_d = nn.Linear(dim, dim)
        self.k_proj_p = nn.Linear(dim, dim)
        self.v_proj_p = nn.Linear(dim, dim)
        self.q_proj_p = nn.Linear(dim, dim)
        self.k_proj_d = nn.Linear(dim, dim)
        self.v_proj_d = nn.Linear(dim, dim)
        self.out_proj_d = nn.Linear(dim, dim)
        self.out_proj_p = nn.Linear(dim, dim)
        self.gate_d = nn.Linear(2 * dim, dim)
        self.gate_p = nn.Linear(2 * dim, dim)
        self.pool_drug = nn.Linear(dim, 1)
        self.pool_prot = nn.Linear(dim, 1)
        self.fc1 = nn.Linear(dim * 3, 256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 64)
        self.ln2 = nn.LayerNorm(64)
        self.fc3 = nn.Linear(64, 1)
        self.drop = nn.Dropout(0.3)
        self.pocket_gate = nn.Linear(64, 1)

    def _attention(self, q, k, v, proj):
        d_k = self.head_dim
        scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        return proj(out)

    def forward(self, drug_nodes, protein_nodes, protein_batch):
        device = drug_nodes.device
        d_seq = drug_nodes.unsqueeze(0)
        p_seq = protein_nodes.unsqueeze(0)

        attended_d = self._attention(
            self.q_proj_d(d_seq), self.k_proj_p(p_seq),
            self.v_proj_p(p_seq), self.out_proj_d
        )
        attended_p = self._attention(
            self.q_proj_p(p_seq), self.k_proj_d(d_seq),
            self.v_proj_d(d_seq), self.out_proj_p
        )

        gate_d = torch.sigmoid(self.gate_d(torch.cat([d_seq, attended_d], dim=-1)))
        enhanced_d = d_seq * gate_d + attended_d * (1 - gate_d)
        gate_p = torch.sigmoid(self.gate_p(torch.cat([p_seq, attended_p], dim=-1)))
        enhanced_p = p_seq * gate_p + attended_p * (1 - gate_p)

        drug_rep = global_add_pool(
            enhanced_d.squeeze(0) * torch.sigmoid(self.pool_drug(enhanced_d.squeeze(0))),
            torch.zeros(drug_nodes.size(0), dtype=torch.long, device=device)
        )

        unique_batches = torch.unique(protein_batch)
        pocket_reps = []
        for b in unique_batches:
            mask = (protein_batch == b)
            nodes = enhanced_p.squeeze(0)[mask]
            gated = nodes * torch.sigmoid(self.pool_prot(nodes))
            pocket_reps.append(gated.mean(dim=0))
        pocket_reps = torch.stack(pocket_reps)

        N = pocket_reps.size(0)
        drug_expanded = drug_rep.expand(N, -1)
        hadamard = drug_expanded * pocket_reps
        combined = torch.cat([drug_expanded, pocket_reps, hadamard], dim=-1)
        h1 = self.drop(F.relu(self.ln1(self.fc1(combined))))
        h2 = self.drop(F.relu(self.ln2(self.fc2(h1))))
        pocket_scores = torch.sigmoid(self.fc3(h2))
        pocket_gate = torch.sigmoid(self.pocket_gate(h2))
        return (pocket_scores * pocket_gate).mean()


class DTIModelV2(nn.Module):
    def __init__(self, conv_type='GCN', latent_dim=31):
        super().__init__()
        self.conv_type = conv_type
        self.latent_dim = latent_dim

        protein_dims = [31] + [latent_dim] * 5
        drug_dims = [74, 70, 65, 60, 55, latent_dim]

        self.protein_encoder = ConvEncoder(conv_type, protein_dims)
        self.drug_encoder = ConvEncoder(conv_type, drug_dims)
        self.decoder = PerPocketCrossAttnDecoder(latent_dim)

    def encode_drug(self, x, edge_index):
        return self.drug_encoder(x, edge_index)

    def encode_protein(self, x, edge_index):
        return self.protein_encoder(x, edge_index)

    def gae_recon_loss(self, z, edge_index, num_nodes):
        pos_score = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-15).mean()
        neg_edge_index = negative_sampling(edge_index, num_nodes=num_nodes,
                                           num_neg_samples=edge_index.size(1))
        neg_score = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=-1)
        neg_loss = -torch.log(1 - torch.sigmoid(neg_score) + 1e-15).mean()
        return pos_loss + neg_loss

    def forward(self, protein_graph, drug_graph):
        protein_feat = self.encode_protein(protein_graph.x, protein_graph.edge_index)
        drug_feat = self.encode_drug(drug_graph.x, drug_graph.edge_index)
        return self.decoder(drug_feat, protein_feat, protein_graph.batch)
