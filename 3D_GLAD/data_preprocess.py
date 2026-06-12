import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from torch_geometric.utils import add_self_loops
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from tqdm import tqdm
import deepchem
from Bio.PDB import PDBList, PDBParser
from scipy.spatial import distance_matrix


PROTEIN_ATOM_SYMBOLS = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'B', 'H']
PROTEIN_ATOM_DEGREES = [0, 1, 2, 3, 4, 5, 6, 7, 8]
PROTEIN_ATOM_HS = [0, 1, 2, 3, 4]
PROTEIN_ATOM_VALENCE = [0, 1, 2, 3, 4, 5]

DRUG_ATOM_SYMBOLS = [
    'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg',
    'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'Se', 'Zn', 'Cu',
    'Ni', 'Co', 'Mn', 'H', 'K', 'Ru', 'Ge', 'Ag', 'Bi', 'Ti',
    'Sn', 'V',
]
DRUG_ATOM_DEGREES = [0, 1, 2, 3, 4, 5]
DRUG_ATOM_HS = [0, 1, 2, 3, 4]
DRUG_ATOM_VALENCE = [0, 1, 2, 3, 4, 5]
DRUG_ATOM_CHARGES = [-2, -1, 0, 1, 2]
DRUG_ATOM_RADICALS = [0, 1, 2, 3, 4]
DRUG_ATOM_HYBRIDIZATION = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]

DRUG_FEAT_DIM = 74
PROTEIN_FEAT_DIM = 31


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def protein_atom_feature(atom):
    return np.array(
        one_of_k_encoding_unk(atom.GetSymbol(), PROTEIN_ATOM_SYMBOLS)
        + one_of_k_encoding_unk(atom.GetDegree(), PROTEIN_ATOM_DEGREES)
        + one_of_k_encoding_unk(atom.GetTotalNumHs(), PROTEIN_ATOM_HS)
        + one_of_k_encoding_unk(atom.GetImplicitValence(), PROTEIN_ATOM_VALENCE)
        + [atom.GetIsAromatic()],
        dtype=np.float32,
    )


def drug_atom_feature(atom):
    symbol = one_of_k_encoding_unk(atom.GetSymbol(), DRUG_ATOM_SYMBOLS)
    degree = one_of_k_encoding_unk(atom.GetDegree(), DRUG_ATOM_DEGREES)
    hs = one_of_k_encoding_unk(atom.GetTotalNumHs(), DRUG_ATOM_HS)
    valence = one_of_k_encoding_unk(atom.GetImplicitValence(), DRUG_ATOM_VALENCE)
    aromatic = [atom.GetIsAromatic()]
    charge = one_of_k_encoding_unk(atom.GetFormalCharge(), DRUG_ATOM_CHARGES)
    radical = one_of_k_encoding_unk(atom.GetNumRadicalElectrons(), DRUG_ATOM_RADICALS)
    hybridization = one_of_k_encoding_unk(atom.GetHybridization(), DRUG_ATOM_HYBRIDIZATION)
    ring = [atom.IsInRing()]
    feat = symbol + degree + hs + valence + aromatic + charge + radical + hybridization + ring
    feat += [0] * (DRUG_FEAT_DIM - len(feat))
    return np.array(feat, dtype=np.float32)


def process_protein_pockets(pdb_file):
    m = Chem.MolFromPDBFile(pdb_file)
    if m is None:
        return None
    if m.GetNumConformers() == 0:
        return None

    am = GetAdjacencyMatrix(m)
    try:
        pk = deepchem.dock.ConvexHullPocketFinder()
        pockets = pk.find_pockets(pdb_file)
    except Exception:
        return None

    if pockets is None or len(pockets) == 0:
        return None

    n_atoms = m.GetNumAtoms()
    conformer = m.GetConformers()[0]
    positions = np.array(conformer.GetPositions())

    pocket_graphs = []
    for bound_box in pockets:
        x_min, x_max = bound_box.x_range
        y_min, y_max = bound_box.y_range
        z_min, z_max = bound_box.z_range

        idxs = []
        for idx in range(n_atoms):
            pos = positions[idx]
            if x_min < pos[0] < x_max and y_min < pos[1] < y_max and z_min < pos[2] < z_max:
                idxs.append(idx)

        if len(idxs) == 0:
            continue

        idxs_arr = np.array(idxs)
        sub_am = am[idxs_arr[:, None], idxs_arr]

        features = np.array([protein_atom_feature(m.GetAtomWithIdx(int(i))) for i in idxs_arr])

        rows, cols = np.where(sub_am > 0)
        edge_mask = rows != cols
        rows, cols = rows[edge_mask], cols[edge_mask]

        if len(rows) == 0:
            edge_index = torch.zeros(2, 0, dtype=torch.long)
        else:
            edge_index = torch.tensor(np.stack([rows, cols]), dtype=torch.long)

        edge_index, _ = add_self_loops(edge_index, num_nodes=len(idxs))
        x = torch.tensor(features, dtype=torch.float)
        pocket_graphs.append(Data(x=x, edge_index=edge_index))

    if len(pocket_graphs) == 0:
        return None

    return Batch.from_data_list(pocket_graphs)


def process_drug(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    features = np.array([drug_atom_feature(atom) for atom in mol.GetAtoms()])
    x = torch.tensor(features, dtype=torch.float)

    edge_list = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_list.append([i, j])
        edge_list.append([j, i])

    if len(edge_list) == 0:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    edge_index, _ = add_self_loops(edge_index, num_nodes=mol.GetNumAtoms())
    return Data(x=x, edge_index=edge_index)


def build_fallback_protein_graph():
    x = torch.zeros(5, PROTEIN_FEAT_DIM)
    for i in range(5):
        x[i, 0] = 1.0
        x[i, 10] = 1.0
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4],
                                [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long)
    edge_index, _ = add_self_loops(edge_index, num_nodes=5)
    return Data(x=x, edge_index=edge_index)


def load_interpretable_dtip(data_dir):
    chem_ids = []
    with open(os.path.join(data_dir, 'chem'), 'r') as f:
        for line in f:
            chem_ids.append(line.strip())

    chem_smiles = []
    with open(os.path.join(data_dir, 'chem.repr'), 'r') as f:
        for line in f:
            chem_smiles.append(line.strip())

    protein_ids = []
    with open(os.path.join(data_dir, 'protein'), 'r') as f:
        for line in f:
            protein_ids.append(line.strip())

    protein_seqs = []
    with open(os.path.join(data_dir, 'protein.repr'), 'r') as f:
        for line in f:
            seq = [int(x) for x in line.strip().split()]
            protein_seqs.append(seq)

    chem_id_to_idx = {cid: i for i, cid in enumerate(chem_ids)}
    protein_id_to_idx = {pid: i for i, pid in enumerate(protein_ids)}

    edges = []
    labels = []
    with open(os.path.join(data_dir, 'edges.pos'), 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            edges.append((parts[1], parts[3]))
            labels.append(1)

    with open(os.path.join(data_dir, 'edges.neg'), 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            edges.append((parts[1], parts[3]))
            labels.append(0)

    return chem_smiles, protein_seqs, chem_id_to_idx, protein_id_to_idx, edges, labels


def load_uniprot_to_pdb(mapping_file):
    mapping = {}
    with open(mapping_file, 'r') as f:
        lines = f.read().strip().split('\n')[1:]
        for line in lines:
            parts = line.split(',')
            if len(parts) >= 2:
                uniprot = parts[0].strip()
                pdb_id = parts[1].strip()[:4]
                mapping[uniprot] = pdb_id
    return mapping


class DTIDataset(Dataset):

    def __init__(self, drug_graphs, protein_graphs, edges, labels):
        self.drug_graphs = drug_graphs
        self.protein_graphs = protein_graphs
        self.edges = edges
        self.labels = labels
        self._fallback_protein = Batch.from_data_list([build_fallback_protein_graph()])
        self._fallback_drug = Data(
            x=torch.zeros(1, DRUG_FEAT_DIM),
            edge_index=torch.zeros(2, 0, dtype=torch.long),
        )

    def __len__(self):
        return len(self.edges)

    def __getitem__(self, idx):
        chem_id, protein_id = self.edges[idx]
        label = self.labels[idx]
        drug_graph = self.drug_graphs.get(chem_id, self._fallback_drug)
        protein_graph = self.protein_graphs.get(protein_id, self._fallback_protein)
        return protein_graph, drug_graph, label


def collate_fn(batch):
    protein_graphs = [item[0] for item in batch]
    drug_graphs = [item[1] for item in batch]
    labels = torch.tensor([item[2] for item in batch], dtype=torch.float)
    return protein_graphs, drug_graphs, labels


def build_datasets(base_dir, cache_dir=None):
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
    os.makedirs(cache_dir, exist_ok=True)

    cache_file = os.path.join(cache_dir, 'attention_data.pkl')
    if os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}")
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        return data['train'], data['test'], data['dev']

    mapping_file = os.path.join(os.path.dirname(__file__), 'uniprot_to_pdb.txt')
    uniprot_to_pdb = load_uniprot_to_pdb(mapping_file)
    pdb_dir = os.path.join(cache_dir, 'pdbs')
    os.makedirs(pdb_dir, exist_ok=True)

    splits = {}
    for split_name in ['train', 'test', 'dev']:
        split_dir = os.path.join(base_dir, split_name)
        data = load_interpretable_dtip(split_dir)
        splits[split_name] = data

    all_smiles = {}
    all_protein_ids = set()
    for split_name, (smiles, seqs, c2i, p2i, edges, labels) in splits.items():
        for cid, idx in c2i.items():
            if cid not in all_smiles:
                all_smiles[cid] = smiles[idx]
        for pid in p2i:
            all_protein_ids.add(pid)

    drug_graphs = {}
    print(f"Building {len(all_smiles)} drug graphs...")
    for cid, smi in tqdm(all_smiles.items()):
        g = process_drug(smi)
        if g is not None:
            drug_graphs[cid] = g
    print(f"  Drug graphs: {len(drug_graphs)}/{len(all_smiles)} successful")

    pdbl = PDBList()
    protein_graphs = {}
    print(f"Building {len(all_protein_ids)} protein pocket graphs...")
    for pid in tqdm(all_protein_ids):
        pdb_id = uniprot_to_pdb.get(pid)
        if pdb_id:
            pdb_file = os.path.join(pdb_dir, f"{pdb_id}.pdb")
            if not os.path.exists(pdb_file):
                try:
                    pdbl.retrieve_pdb_file(pdb_id, pdir=pdb_dir, file_format='pdb')
                    ent_file = os.path.join(pdb_dir, f"pdb{pdb_id.lower()}.ent")
                    if os.path.exists(ent_file):
                        os.rename(ent_file, pdb_file)
                except Exception:
                    pass

            if os.path.exists(pdb_file):
                try:
                    g = process_protein_pockets(pdb_file)
                    if g is not None:
                        protein_graphs[pid] = g
                        continue
                except Exception:
                    pass

        protein_graphs[pid] = Batch.from_data_list([build_fallback_protein_graph()])

    n_pdb = sum(1 for pid in all_protein_ids if uniprot_to_pdb.get(pid))
    n_success = sum(
        1 for pid in all_protein_ids
        if pid in protein_graphs
        and protein_graphs[pid].num_graphs > 0
        and uniprot_to_pdb.get(pid)
    )
    print(f"  PDB proteins: {n_pdb} mapped, {n_success} with pocket graphs")

    datasets = {}
    for split_name, (smiles, seqs, c2i, p2i, edges, labels) in splits.items():
        datasets[split_name] = DTIDataset(drug_graphs, protein_graphs, edges, labels)
        print(f"  {split_name}: {len(datasets[split_name])} samples")

    with open(cache_file, 'wb') as f:
        pickle.dump(datasets, f)

    return datasets['train'], datasets['test'], datasets['dev']


def build_dataloaders(base_dir, batch_size=256, num_workers=0):
    train_dataset, test_dataset, dev_dataset = build_datasets(base_dir)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers)
    dev_loader = torch.utils.data.DataLoader(
        dev_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers)
    return train_loader, test_loader, dev_loader
