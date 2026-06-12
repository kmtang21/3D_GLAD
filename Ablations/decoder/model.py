import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.utils import negative_sampling


class GCNEncoder(nn.Module):
    def __init__(self, layer_dims):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_dims) - 1):
            self.layers.append(GCNConv(layer_dims[i], layer_dims[i + 1]))

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = F.relu(layer(x, edge_index))
        return x


class GatedPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate_nn = nn.Linear(dim, 1)

    def forward(self, x, batch=None):
        gate = torch.sigmoid(self.gate_nn(x))
        return global_add_pool(x * gate, batch)


class PerPocketMLPDecoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        total = dim * 4
        self.fc1 = nn.Linear(total, 256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)
        self.fc3 = nn.Linear(128, 64)
        self.ln3 = nn.LayerNorm(64)
        self.fc4 = nn.Linear(64, 1)
        self.drop = nn.Dropout(0.3)
        self.pocket_gate = nn.Linear(64, 1)

    def forward(self, drug_rep, pocket_reps):
        N = pocket_reps.size(0)
        drug_expanded = drug_rep.expand(N, -1)
        hadamard = drug_expanded * pocket_reps
        diff = torch.abs(drug_expanded - pocket_reps)
        combined = torch.cat([drug_expanded, pocket_reps, hadamard, diff], dim=-1)
        h1 = self.drop(F.relu(self.ln1(self.fc1(combined))))
        h2 = self.drop(F.relu(self.ln2(self.fc2(h1))))
        h3 = self.drop(F.relu(self.ln3(self.fc3(h2))))
        pocket_scores = self.fc4(h3)
        pocket_gate = torch.sigmoid(self.pocket_gate(h3))
        weighted = pocket_scores * pocket_gate
        return torch.sigmoid(weighted.mean())


class PerPocketBilinearDecoder(nn.Module):
    def __init__(self, dim, rank=32):
        super().__init__()
        self.drug_proj = nn.Linear(dim, rank)
        self.prot_proj = nn.Linear(dim, rank)
        self.fc1 = nn.Linear(rank, 128)
        self.ln1 = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, 64)
        self.ln2 = nn.LayerNorm(64)
        self.fc3 = nn.Linear(64, 1)
        self.drop = nn.Dropout(0.3)
        self.pocket_attn = nn.Linear(rank, 1)

    def forward(self, drug_rep, pocket_reps):
        dp = self.drug_proj(drug_rep)
        pp = self.prot_proj(pocket_reps)
        bilinear = dp * pp
        h1 = self.drop(F.relu(self.ln1(self.fc1(bilinear))))
        h2 = self.drop(F.relu(self.ln2(self.fc2(h1))))
        pocket_scores = torch.sigmoid(self.fc3(h2))
        attn = F.softmax(self.pocket_attn(bilinear), dim=0)
        return (pocket_scores * attn).sum()


class PerPocketNTNDecoder(nn.Module):
    def __init__(self, dim, num_slices=16):
        super().__init__()
        self.K = num_slices
        self.W = nn.ParameterList(
            [nn.Parameter(torch.randn(dim, dim) * 0.01)
             for _ in range(num_slices)]
        )
        self.V = nn.Linear(dim * 2, num_slices)
        self.b = nn.Parameter(torch.zeros(num_slices))
        self.fc1 = nn.Linear(num_slices, 64)
        self.ln1 = nn.LayerNorm(64)
        self.fc2 = nn.Linear(64, 1)
        self.drop = nn.Dropout(0.3)
        self.pocket_attn = nn.Linear(num_slices, 1)

    def forward(self, drug_rep, pocket_reps):
        N = pocket_reps.size(0)
        drug_expanded = drug_rep.expand(N, -1)
        slices = []
        for k in range(self.K):
            bk = (drug_expanded @ self.W[k] * pocket_reps).sum(dim=-1, keepdim=True)
            slices.append(bk)
        bilinear = torch.cat(slices, dim=-1)
        linear = self.V(torch.cat([drug_expanded, pocket_reps], dim=-1))
        h = torch.tanh(bilinear + linear + self.b)
        h = self.drop(F.relu(self.ln1(self.fc1(h))))
        pocket_scores = torch.sigmoid(self.fc2(h))
        attn = F.softmax(self.pocket_attn(bilinear), dim=0)
        return (pocket_scores * attn).sum()


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
        unique_batches = torch.unique(protein_batch)

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


class PerPocketNodeBilinearDecoder(nn.Module):
    def __init__(self, dim, proj_dim=64):
        super().__init__()
        self.drug_proj = nn.Linear(dim, proj_dim)
        self.prot_proj = nn.Linear(dim, proj_dim)
        self.pool_drug = nn.Linear(proj_dim, 1)
        self.pool_prot = nn.Linear(proj_dim, 1)
        self.node_attn = nn.Linear(proj_dim, 1)

        self.fc1 = nn.Linear(proj_dim * 3, 256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 64)
        self.ln2 = nn.LayerNorm(64)
        self.fc3 = nn.Linear(64, 1)
        self.drop = nn.Dropout(0.3)
        self.pocket_gate = nn.Linear(64, 1)

    def forward(self, drug_nodes, protein_nodes, protein_batch):
        device = drug_nodes.device
        d_proj = self.drug_proj(drug_nodes)
        p_proj = self.prot_proj(protein_nodes)

        interaction_map = torch.matmul(d_proj, p_proj.t())
        p_enhanced = p_proj + torch.matmul(interaction_map.t(), d_proj) * 0.1

        d_gated = d_proj * torch.sigmoid(self.pool_drug(d_proj))
        drug_rep = d_gated.mean(dim=0, keepdim=True)

        unique_batches = torch.unique(protein_batch)
        pocket_reps = []
        for b in unique_batches:
            mask = (protein_batch == b)
            nodes = p_enhanced[mask]
            attn_w = F.softmax(self.node_attn(nodes), dim=0)
            pocket_reps.append((nodes * attn_w).sum(dim=0))
        pocket_reps = torch.stack(pocket_reps)

        N = pocket_reps.size(0)
        drug_expanded = drug_rep.expand(N, -1)
        combined = torch.cat([drug_expanded, pocket_reps, drug_expanded * pocket_reps], dim=-1)
        h1 = self.drop(F.relu(self.ln1(self.fc1(combined))))
        h2 = self.drop(F.relu(self.ln2(self.fc2(h1))))
        pocket_scores = torch.sigmoid(self.fc3(h2))
        pocket_gate = torch.sigmoid(self.pocket_gate(h2))
        return (pocket_scores * pocket_gate).mean()


DECODERS = {
    'mlp': PerPocketMLPDecoder,
    'bilinear': PerPocketBilinearDecoder,
    'ntn': PerPocketNTNDecoder,
    'crossattn': PerPocketCrossAttnDecoder,
    'nodebilinear': PerPocketNodeBilinearDecoder,
}


class DTIPredModel(nn.Module):
    def __init__(self, decoder_type='mlp', latent_dim=31):
        super().__init__()
        self.decoder_type = decoder_type
        self.latent_dim = latent_dim

        protein_dims = [31] + [latent_dim] * 5
        drug_dims = [74, 70, 65, 60, 55, latent_dim]

        self.protein_encoder = GCNEncoder(protein_dims)
        self.drug_encoder = GCNEncoder(drug_dims)

        if decoder_type in ('crossattn', 'nodebilinear'):
            self.decoder = DECODERS[decoder_type](latent_dim)
            self.protein_pool = None
            self.drug_pool = None
        else:
            self.decoder = DECODERS[decoder_type](latent_dim)
            self.protein_pool = GatedPooling(latent_dim)
            self.drug_pool = GatedPooling(latent_dim)

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
        device = protein_graph.x.device
        protein_feat = self.encode_protein(protein_graph.x, protein_graph.edge_index)
        drug_feat = self.encode_drug(drug_graph.x, drug_graph.edge_index)

        if self.decoder_type in ('crossattn', 'nodebilinear'):
            return self.decoder(drug_feat, protein_feat, protein_graph.batch)

        drug_rep = self.drug_pool(drug_feat)
        pocket_reps = self.protein_pool(protein_feat, protein_graph.batch)
        return self.decoder(drug_rep, pocket_reps)
