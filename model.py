import streamlit as st
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from transformers import AutoTokenizer, AutoModel
from rdkit import Chem
# from rdkit.Chem import AllChem, Draw
# from rdkit.Chem.Draw import rdMolDraw2D
# from PIL import Image
import io

tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
model_chemberta = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

def smiles_to_embedding(smiles):
    inputs = tokenizer(smiles, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model_chemberta(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()

def smiles_to_fingerprint(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return torch.tensor(fingerprint, dtype=torch.float32)

def create_data_object(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    molecule_embedding = smiles_to_embedding(smiles)
    fingerprint = smiles_to_fingerprint(smiles)
    if fingerprint is None:
        return None

    combined_features = torch.cat([molecule_embedding, fingerprint])
    atom_features = combined_features.repeat(mol.GetNumAtoms(), 1)

    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])
        edge_attr.extend([1.0, 1.0])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32).view(-1, 1)

    return Data(x=atom_features, edge_index=edge_index, edge_attr=edge_attr)

class GNN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=128, dropout=0.3):
        super(GNN, self).__init__()
        self.transformer1 = TransformerConv(num_features, hidden_dim, heads=4, dropout=dropout)
        self.transformer2 = TransformerConv(hidden_dim * 4, hidden_dim, heads=4, concat=False, dropout=dropout)
        self.fc_out = torch.nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, batch, edge_attr=None):
        x = F.elu(self.transformer1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.transformer2(x, edge_index))
        x = global_mean_pool(x, batch)
        return F.log_softmax(self.fc_out(x), dim=1)
