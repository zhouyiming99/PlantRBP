#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import gc
import sys
import json
import pickle
import random
import warnings
import subprocess
import argparse
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from scipy.stats import fisher_exact
from tqdm.auto import tqdm

from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data, Batch
from torch_geometric.utils import softmax as pyg_softmax

try:
    import esm
except ImportError:
    raise ImportError("Please install: pip install fair-esm")

warnings.filterwarnings('ignore')




DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()




class EdgeWeightedGATConv(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4, dropout=0.3, concat=True):
        super().__init__()
        self.gat = GATv2Conv(
            in_dim, out_dim, heads=heads,
            dropout=dropout, concat=concat, edge_dim=1
        )
        out_features = out_dim * heads if concat else out_dim
        self.norm    = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr=None):
        out = self.gat(x, edge_index, edge_attr=edge_attr)
        out = self.norm(out)
        return F.elu(self.dropout(out))


class MultiScalePooling(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.att = nn.Sequential(
            nn.Linear(in_dim, in_dim // 4),
            nn.Tanh(),
            nn.Linear(in_dim // 4, 1)
        )
        self.gate = nn.Sequential(
            nn.Linear(in_dim * 3, in_dim),
            nn.Sigmoid()
        )

    def forward(self, x, batch):
        x         = x.float()
        mean_pool = global_mean_pool(x, batch)
        max_pool  = global_max_pool(x, batch)
        scores    = self.att(x).squeeze(-1)
        weights   = pyg_softmax(scores, batch)
        att_pool  = global_add_pool(x * weights.unsqueeze(-1), batch)
        concat    = torch.cat([mean_pool, max_pool, att_pool], dim=-1)
        gate      = self.gate(concat)
        fused     = gate * mean_pool + (1 - gate) * (max_pool + att_pool) / 2
        return torch.cat([fused, att_pool], dim=-1)


class BalancedGATModel(nn.Module):
    """Full PPI + sequence fusion model (architecture identical to training script)"""

    def __init__(
        self,
        node_dim     : int,
        seq_dim      : int,
        hidden_dim   : int   = 384,
        gat_hidden   : int   = 128,
        gat_heads    : int   = 4,
        num_layers   : int   = 3,
        dropout      : float = 0.5,
        use_edge_attr: bool  = True,
    ):
        super().__init__()

        
        self.node_proj = nn.Sequential(
            nn.Linear(node_dim, gat_hidden * gat_heads),
            nn.LayerNorm(gat_hidden * gat_heads),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gat_layers = nn.ModuleList()
        current_dim = gat_hidden * gat_heads
        for i in range(num_layers):
            concat = (i != num_layers - 1)
            self.gat_layers.append(
                EdgeWeightedGATConv(
                    current_dim, gat_hidden,
                    heads=gat_heads, dropout=dropout, concat=concat
                )
            )
            current_dim = gat_hidden * gat_heads if concat else gat_hidden

        self.pooling      = MultiScalePooling(current_dim)
        pool_out_dim      = current_dim * 2
        self.graph_encoder = nn.Sequential(
            nn.Linear(pool_out_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.aux_classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
        )

        
        self.seq_encoder = nn.Sequential(
            nn.Linear(seq_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )

        
        fusion_dim = hidden_dim // 2 + hidden_dim // 2
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim // 2, num_heads=4, dropout=dropout, batch_first=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim + hidden_dim // 2, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.feature_dropout = nn.Dropout(0.15)
        self.use_edge_attr   = use_edge_attr

    def forward(self, graph_data, seq_features, training=False):
        x          = graph_data.x
        edge_index = graph_data.edge_index
        edge_attr  = graph_data.edge_attr if self.use_edge_attr else None
        batch      = graph_data.batch

        if training and self.training:
            seq_features = self.feature_dropout(seq_features)

        x = self.node_proj(x)
        for gat_layer in self.gat_layers:
            x = gat_layer(x, edge_index, edge_attr)
        graph_pool = self.pooling(x, batch)
        graph_out  = self.graph_encoder(graph_pool)
        aux_out    = self.aux_classifier(graph_out).squeeze(-1)
        seq_out    = self.seq_encoder(seq_features.float())

        graph_q  = graph_out.unsqueeze(1)
        seq_kv   = seq_out.unsqueeze(1)
        cross, _ = self.cross_attention(graph_q, seq_kv, seq_kv)
        cross    = cross.squeeze(1)

        combined = torch.cat([graph_out, seq_out, cross], dim=-1)
        main_out = self.classifier(combined).squeeze(-1)
        return main_out, aux_out


class SeqOnlyModel(nn.Module):
    """
    Sequence-only model.
    Extracts seq_encoder parameters from BalancedGATModel weights.
    """

    def __init__(self, seq_dim: int, hidden_dim: int = 384, dropout: float = 0.5):
        super().__init__()
        self.seq_encoder = nn.Sequential(
            nn.Linear(seq_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
        )

    def load_from_full_model(self, state_dict: dict):
        seq_keys = {
            k.replace('seq_encoder.', ''): v
            for k, v in state_dict.items()
            if k.startswith('seq_encoder.')
        }
        self.seq_encoder.load_state_dict(seq_keys)
        print("    [OK] seq_encoder weights loaded from full model")

    def forward(self, seq_features):
        seq_out = self.seq_encoder(seq_features.float())
        return self.classifier(seq_out).squeeze(-1)



def load_rbp_ids(filepath: str) -> Set[str]:
    with open(filepath, 'r') as f:
        return set(line.strip() for line in f if line.strip())


def load_fasta(filepath: str) -> Dict[str, str]:
    sequences = {}
    current_id, current_seq = None, []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id and current_seq:
                    sequences[current_id] = ''.join(current_seq)
                current_id  = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
        if current_id and current_seq:
            sequences[current_id] = ''.join(current_seq)
    return sequences


def load_ppi(
    filepath  : str,
    sequences : Dict[str, str],
    min_score : float = 0.60
) -> Dict[str, List[Tuple[str, float]]]:
    adjacency = defaultdict(list)
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                p1, p2 = parts[0], parts[1]
                score  = float(parts[2]) if len(parts) > 2 else 1.0
                if (p1 in sequences and p2 in sequences
                        and score >= min_score and p1 != p2):
                    adjacency[p1].append((p2, score))
                    adjacency[p2].append((p1, score))
    for k in adjacency:
        seen = {}
        for neighbor, score in adjacency[k]:
            if neighbor not in seen or seen[neighbor] < score:
                seen[neighbor] = score
        adjacency[k] = sorted(seen.items(), key=lambda x: x[1], reverse=True)
    return dict(adjacency)




class FeatureExtractor:
    def __init__(self):
        self.amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        self.aa_to_idx   = {aa: i for i, aa in enumerate(self.amino_acids)}
        self.rbp_motifs  = [
            'RGG', 'GGG', 'RRM', 'KH', 'SR', 'RS', 'RR', 'KK', 'GG',
            'RGR', 'KGK', 'YGG', 'FGG', 'DEAD', 'DEAH', 'GGGG', 'RGGR',
            'GAR', 'RG', 'GR', 'KG', 'GK'
        ]

    def extract(self, sequence: str) -> np.ndarray:
        seq = ''.join(aa for aa in sequence.upper() if aa in self.amino_acids)
        if len(seq) < 5:
            seq = seq + 'A' * (5 - len(seq))
        features = []
        aa_comp  = np.zeros(20)
        for aa in seq:
            aa_comp[self.aa_to_idx[aa]] += 1
        features.extend(aa_comp / len(seq))
        for motif in self.rbp_motifs:
            features.append(seq.count(motif) / len(seq))
        features.append(sum(1 for aa in seq if aa in 'RKH') / len(seq))
        features.append(sum(1 for aa in seq if aa in 'DE')  / len(seq))
        features.append(len(seq) / 1000)
        features.append(np.log1p(len(seq)) / 10)
        charges = [
            1 if aa in 'RKH' else (-1 if aa in 'DE' else 0)
            for aa in seq
        ]
        features.append(np.std(charges))
        return np.array(features, dtype=np.float32)


class GraphFeatures:
    def __init__(self, adjacency: Dict):
        self.adjacency  = adjacency
        self.degrees    = {n: len(v) for n, v in adjacency.items()}
        self.max_degree = max(self.degrees.values()) if self.degrees else 1

    def extract(self, node: str) -> np.ndarray:
        degree   = self.degrees.get(node, 0)
        features = [
            degree / 100.0,
            np.log1p(degree) / 5.0,
            degree / self.max_degree
        ]
        if node in self.adjacency and self.adjacency[node]:
            ws = [s for _, s in self.adjacency[node]]
            features.extend([np.mean(ws), np.max(ws)])
        else:
            features.extend([0.0, 0.0])
        return np.array(features, dtype=np.float32)


def extract_esm_features(
    protein_ids     : List[str],
    sequences_dict  : Dict[str, str],
    device          : torch.device,
    cache_dir       : str = './predict_cache',
    train_cache_path: Optional[str] = None,
    batch_size      : int = 6,
) -> Dict[str, torch.Tensor]:
    """
    Extract ESM-2 650M features.
    Priority: training cache (train_cache_path) -> prediction cache -> compute on-the-fly.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cached: Dict[str, torch.Tensor] = {}

    
    if train_cache_path and os.path.exists(train_cache_path):
        print(f"  Reusing training cache: {train_cache_path}")
        with open(train_cache_path, 'rb') as f:
            cached = pickle.load(f)

    
    predict_cache = os.path.join(cache_dir, 'esm_predict.pkl')
    if os.path.exists(predict_cache):
        with open(predict_cache, 'rb') as f:
            cached.update(pickle.load(f))

    to_compute = [pid for pid in protein_ids if pid not in cached]

    if to_compute:
        print(f"  New proteins requiring ESM feature computation: {len(to_compute)}")
        print("  Loading ESM-2 650M model...")
        esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        batch_converter      = alphabet.get_batch_converter()
        esm_model            = esm_model.to(device).eval()
        new_features: Dict[str, torch.Tensor] = {}

        for i in tqdm(range(0, len(to_compute), batch_size), desc="  ESM features"):
            batch_pids = to_compute[i:i + batch_size]
            batch_data = []
            for pid in batch_pids:
                raw_seq = sequences_dict.get(pid, 'A')
                seq = ''.join(
                    aa for aa in raw_seq.upper()
                    if aa in 'ACDEFGHIKLMNPQRSTVWY'
                )
                seq = seq[:1022] if len(seq) > 1022 else (seq or 'A')
                batch_data.append((pid, seq))

            try:
                _, batch_strs, batch_tokens = batch_converter(batch_data)
                batch_tokens = batch_tokens.to(device)
                with torch.no_grad():
                    results    = esm_model(
                        batch_tokens, repr_layers=[33], return_contacts=False
                    )
                    token_repr = results["representations"][33]

                for j, pid in enumerate(batch_pids):
                    seq_len      = len(batch_strs[j])
                    seq_feat     = token_repr[j, 1:seq_len + 1]
                    mean_pool    = seq_feat.mean(dim=0)
                    max_pool     = seq_feat.max(dim=0)[0]
                    norms        = seq_feat.norm(dim=-1, keepdim=True)
                    weights_w    = F.softmax(norms.squeeze(-1), dim=0).unsqueeze(-1)
                    weighted_pool= (seq_feat * weights_w).sum(dim=0)
                    new_features[pid] = torch.cat(
                        [mean_pool, max_pool, weighted_pool], dim=-1
                    ).cpu()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    for pid in batch_pids:
                        new_features[pid] = torch.zeros(1280 * 3)
                else:
                    raise

        
        with open(predict_cache, 'wb') as f:
            pickle.dump(new_features, f)
        cached.update(new_features)

        del esm_model, batch_converter, alphabet
        clear_memory()

    return {pid: cached[pid] for pid in protein_ids}


def prepare_features(
    protein_ids     : List[str],
    sequences_dict  : Dict[str, str],
    cache_dir       : str = './predict_cache',
    train_cache_path: Optional[str] = None,
    model_dir       : str = './models',
    esm_batch_size  : int = 6,
) -> Tuple[Dict[str, torch.Tensor], int]:
    
    print("  Extracting ESM-2 features...")
    esm_feats = extract_esm_features(
        protein_ids, sequences_dict, DEVICE,
        cache_dir=cache_dir,
        train_cache_path=train_cache_path,
        batch_size=esm_batch_size,
    )

    
    print("  Extracting hand-crafted sequence features...")
    feat_extractor = FeatureExtractor()
    basic_feats = {
        p: feat_extractor.extract(sequences_dict.get(p, 'A'))
        for p in tqdm(protein_ids, desc="  Hand-crafted features", leave=False)
    }

    
    scaler_path   = os.path.join(model_dir, 'scaler.pkl')
    all_basic_arr = np.array([basic_feats[p] for p in protein_ids])
    if os.path.exists(scaler_path):
        print("  Loading training-set normalization parameters...")
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        all_basic_norm = scaler.transform(all_basic_arr)
    else:
        print("  WARNING: scaler.pkl not found, normalizing with current data")
        scaler         = RobustScaler()
        all_basic_norm = scaler.fit_transform(all_basic_arr)

    
    seq_features: Dict[str, torch.Tensor] = {}
    for i, p in enumerate(protein_ids):
        seq_features[p] = torch.cat([
            esm_feats[p],
            torch.tensor(all_basic_norm[i], dtype=torch.float32)
        ])

    seq_dim = seq_features[protein_ids[0]].shape[0]
    print(f"  Feature dimension: {seq_dim}")
    return seq_features, seq_dim


def build_subgraph(
    target              : str,
    adjacency           : Dict,
    node_features       : Dict[str, torch.Tensor],
    graph_feat_extractor: GraphFeatures,
    max_neighbors       : int = 40,
) -> Data:
    neighbors_with_scores = adjacency.get(target, [])[:max_neighbors]
    neighbors      = [n for n, _ in neighbors_with_scores]
    neighbor_weights = {n: s for n, s in neighbors_with_scores}
    nodes          = [target] + neighbors
    node_to_idx    = {n: i for i, n in enumerate(nodes)}

    feat_list = []
    for i, node in enumerate(nodes):
        base_feat  = node_features.get(node, node_features[target])
        graph_feat = graph_feat_extractor.extract(node)
        is_target  = 1.0 if i == 0 else 0.0
        edge_weight= neighbor_weights.get(node, 0.5)
        pos_feat   = torch.tensor([is_target, edge_weight], dtype=torch.float32)
        feat_list.append(torch.cat([base_feat, torch.tensor(graph_feat), pos_feat]))

    x = torch.stack(feat_list)
    edges, edge_weights = [], []
    for i, n1 in enumerate(nodes):
        for n2, score in adjacency.get(n1, []):
            if n2 in node_to_idx:
                edges.append([i, node_to_idx[n2]])
                edge_weights.append(score)
        edges.append([i, i])
        edge_weights.append(1.0)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr  = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(-1)
    return Data(
        x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(nodes)
    )




class PPIDataset(Dataset):
    def __init__(self, protein_ids, graphs, seq_features):
        self.protein_ids  = protein_ids
        self.graphs       = graphs
        self.seq_features = seq_features

    def __len__(self):
        return len(self.protein_ids)

    def __getitem__(self, idx):
        pid = self.protein_ids[idx]
        return {
            'protein_id'  : pid,
            'graph'       : self.graphs[pid],
            'seq_features': self.seq_features[pid],
        }


class SeqDataset(Dataset):
    def __init__(self, protein_ids, seq_features):
        self.protein_ids  = protein_ids
        self.seq_features = seq_features

    def __len__(self):
        return len(self.protein_ids)

    def __getitem__(self, idx):
        pid = self.protein_ids[idx]
        return {'protein_id': pid, 'seq_features': self.seq_features[pid]}


def collate_ppi(batch):
    return {
        'graph'       : Batch.from_data_list([b['graph'] for b in batch]),
        'seq_features': torch.stack([b['seq_features'] for b in batch]),
        'protein_ids' : [b['protein_id'] for b in batch],
    }


def collate_seq(batch):
    return {
        'seq_features': torch.stack([b['seq_features'] for b in batch]),
        'protein_ids' : [b['protein_id'] for b in batch],
    }




def _make_result_df(
    protein_ids: List[str],
    preds      : np.ndarray,
    threshold  : float,
    mode       : str,
    adjacency  : Optional[Dict] = None,
) -> pd.DataFrame:
    data = {
        'protein_id': protein_ids,
        'rbp_score' : preds,
        'prediction': ['RBP' if p >= threshold else 'non-RBP' for p in preds],
        'confidence': [
            'High'   if abs(p - 0.5) > 0.3  else
            'Medium' if abs(p - 0.5) > 0.15 else 'Low'
            for p in preds
        ],
        'mode': mode,
    }
    if adjacency is not None:
        data['ppi_degree'] = [len(adjacency.get(p, [])) for p in protein_ids]
    return pd.DataFrame(data).sort_values('rbp_score', ascending=False)


def print_summary(df: pd.DataFrame, output_csv: str, threshold: float):
    rbp_n = (df['prediction'] == 'RBP').sum()
    total = len(df)
    print(f"\n{'='*55}")
    print(f"Prediction complete!")
    print(f"  Mode:              {df['mode'].iloc[0]}")
    print(f"  Total proteins:    {total}")
    print(f"  Predicted as RBP:  {rbp_n} ({rbp_n/total*100:.1f}%)")
    print(f"  Predicted non-RBP: {total - rbp_n}")
    print(f"  Classification threshold: {threshold}")
    print(f"  Output file:       {output_csv}")
    print(f"{'='*55}")
    cols = ['protein_id', 'rbp_score', 'prediction', 'confidence']
    if 'ppi_degree' in df.columns:
        cols.append('ppi_degree')
    print("\nTop 10 predicted RBPs:")
    print(df[cols].head(10).to_string(index=False))




def predict_seq_mode(
    query_ids       : List[str],
    sequences_dict  : Dict[str, str],
    model_dir       : str,
    n_folds         : int   = 5,
    use_tta         : bool  = True,
    threshold       : float = 0.5,
    cache_dir       : str   = './predict_cache',
    train_cache_path: Optional[str] = None,
    esm_batch_size  : int   = 6,
) -> pd.DataFrame:
    print("\n" + "="*50)
    print("Mode 1: Sequence-only prediction")
    print("="*50)

    print("\n[1/3] Feature extraction...")
    seq_features, seq_dim = prepare_features(
        query_ids, sequences_dict,
        cache_dir=cache_dir,
        train_cache_path=train_cache_path,
        model_dir=model_dir,
        esm_batch_size=esm_batch_size,
    )

    dataset = SeqDataset(query_ids, seq_features)
    loader  = DataLoader(
        dataset, batch_size=64, shuffle=False,
        collate_fn=collate_seq, num_workers=0
    )

    print("\n[2/3] Model inference (5-fold ensemble)...")
    fold_preds = []

    for fold_i in range(1, n_folds + 1):
        model_path = os.path.join(model_dir, f'balanced_fold_{fold_i}.pt')
        if not os.path.exists(model_path):
            print(f"  WARNING: fold_{fold_i} not found, skipping")
            continue

        print(f"  Loading Fold {fold_i}...")
        state_dict = torch.load(model_path, map_location=DEVICE)

        model = SeqOnlyModel(seq_dim=seq_dim, hidden_dim=384, dropout=0.5).to(DEVICE)
        model.load_from_full_model(state_dict)
        model.eval()

        preds_this_fold = []
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"  Fold {fold_i}", leave=False):
                seq = batch['seq_features'].to(DEVICE)

                if use_tta:
                    ps = [torch.sigmoid(model(seq).float())]
                    for _ in range(8):
                        noisy = seq + torch.randn_like(seq) * 0.02
                        ps.append(torch.sigmoid(model(noisy).float()))
                    p = torch.stack(ps).mean(dim=0)
                else:
                    p = torch.sigmoid(model(seq).float())

                preds_this_fold.extend(p.cpu().numpy())

        fold_preds.append(preds_this_fold)

    final_preds = np.mean(fold_preds, axis=0)

    print("\n[3/3] Saving results...")
    return _make_result_df(query_ids, final_preds, threshold, mode='seq_only')




def predict_ppi_mode(
    query_ids       : List[str],
    sequences_dict  : Dict[str, str],
    ppi_file        : str,
    background_fasta: str,
    model_dir       : str,
    n_folds         : int   = 5,
    use_tta         : bool  = True,
    threshold       : float = 0.5,
    cache_dir       : str   = './predict_cache',
    train_cache_path: Optional[str] = None,
    esm_batch_size  : int   = 6,
) -> pd.DataFrame:
    print("\n" + "="*50)
    print("Mode 2: PPI + sequence prediction")
    print("="*50)

    
    print("\n[1/5] Loading background proteome...")
    bg_seqs  = load_fasta(background_fasta)
    all_seqs = {**bg_seqs, **sequences_dict}
    print(f"  Background proteins: {len(bg_seqs)}, query proteins: {len(query_ids)}")

    
    print("\n[2/5] Building PPI network...")
    adjacency = load_ppi(ppi_file, all_seqs, min_score=0.60)
    connected = sum(1 for p in query_ids
                    if p in adjacency and len(adjacency[p]) >= 1)
    isolated  = len(query_ids) - connected
    print(f"  With PPI connections: {connected} | Isolated nodes: {isolated}")
    if isolated > 0:
        print(f"  WARNING: {isolated} proteins have no PPI connections; isolated graphs (self-loop only) will be used")
    for pid in query_ids:
        if pid not in adjacency:
            adjacency[pid] = []

    
    print("\n[3/5] Feature extraction...")
    neighbor_ids = set()
    for pid in query_ids:
        for n, _ in adjacency.get(pid, [])[:40]:
            neighbor_ids.add(n)
    all_needed = list(set(query_ids) | neighbor_ids)
    print(f"  Proteins requiring features (including neighbors): {len(all_needed)}")

    seq_features, seq_dim = prepare_features(
        all_needed, all_seqs,
        cache_dir=cache_dir,
        train_cache_path=train_cache_path,
        model_dir=model_dir,
        esm_batch_size=esm_batch_size,
    )

    
    print("\n[4/5] Building protein subgraphs...")
    graph_feat_extractor = GraphFeatures(adjacency)
    graphs = {}
    for pid in tqdm(query_ids, desc="  Building graphs"):
        graphs[pid] = build_subgraph(
            pid, adjacency, seq_features, graph_feat_extractor, max_neighbors=40
        )
    node_dim = graphs[query_ids[0]].x.shape[1]
    print(f"  Node feature dimension: {node_dim}")

    
    print("\n[5/5] Model inference (5-fold ensemble)...")
    dataset = PPIDataset(query_ids, graphs, seq_features)
    loader  = DataLoader(
        dataset, batch_size=48, shuffle=False,
        collate_fn=collate_ppi, num_workers=0
    )

    fold_preds = []

    for fold_i in range(1, n_folds + 1):
        model_path = os.path.join(model_dir, f'balanced_fold_{fold_i}.pt')
        if not os.path.exists(model_path):
            print(f"  WARNING: fold_{fold_i} not found, skipping")
            continue

        model = BalancedGATModel(
            node_dim=node_dim, seq_dim=seq_dim,
            hidden_dim=384, gat_hidden=128, gat_heads=4,
            num_layers=3, dropout=0.5, use_edge_attr=True
        ).to(DEVICE)

        state = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state)
        model.eval()

        preds_this_fold = []
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"  Fold {fold_i}", leave=False):
                graph = batch['graph'].to(DEVICE)
                seq   = batch['seq_features'].to(DEVICE)

                if use_tta:
                    ps = [torch.sigmoid(
                        model(graph, seq, training=False)[0].float()
                    )]
                    for _ in range(8):
                        noisy = seq + torch.randn_like(seq) * 0.02
                        out, _ = model(graph, noisy, training=False)
                        ps.append(torch.sigmoid(out.float()))
                    p = torch.stack(ps).mean(dim=0)
                else:
                    p = torch.sigmoid(
                        model(graph, seq, training=False)[0].float()
                    )
                preds_this_fold.extend(p.cpu().numpy())

        fold_preds.append(preds_this_fold)

    final_preds = np.mean(fold_preds, axis=0)
    return _make_result_df(
        query_ids, final_preds, threshold,
        mode='ppi+seq', adjacency=adjacency
    )




class HMMERParser:
    def __init__(
        self,
        evalue_thr  : float = 1e-5,
        score_thr   : float = 20.0,
        coverage_thr: float = 0.5,
    ):
        self.evalue_thr   = evalue_thr
        self.score_thr    = score_thr
        self.coverage_thr = coverage_thr

    def parse_domtblout(self, filepath: str) -> pd.DataFrame:
        records = []
        with open(filepath, 'r') as fh:
            for line in fh:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.split()
                if len(parts) < 22:
                    continue
                try:
                    hmm_len = int(parts[2])
                    records.append({
                        'hmm_name'      : parts[0],
                        'protein_id'    : parts[3],
                        'protein_length': int(parts[5]),
                        'domain_ievalue': float(parts[12]),
                        'domain_score'  : float(parts[13]),
                        'hmm_from'      : int(parts[15]),
                        'hmm_to'        : int(parts[16]),
                        'ali_from'      : int(parts[17]),
                        'ali_to'        : int(parts[18]),
                        'hmm_coverage'  : (
                            int(parts[16]) - int(parts[15]) + 1
                        ) / hmm_len,
                    })
                except (ValueError, IndexError):
                    continue
        df = pd.DataFrame(records)
        print(f"  HMMER raw hits: {len(df)} rows, {df['protein_id'].nunique()} proteins")
        return df

    def filter_hits(self, df: pd.DataFrame) -> pd.DataFrame:
        mask = (
            (df['domain_ievalue'] <= self.evalue_thr) &
            (df['domain_score']   >= self.score_thr)  &
            (df['hmm_coverage']   >= self.coverage_thr)
        )
        out = df[mask].copy()
        print(f"  After filtering: {len(out)} rows, {out['protein_id'].nunique()} proteins")
        return out

    def get_covered_regions(
        self, filtered_df: pd.DataFrame
    ) -> Dict[str, List[Tuple[int, int, str]]]:
        covered: Dict[str, List] = defaultdict(list)
        for _, row in filtered_df.iterrows():
            covered[row['protein_id']].append((
                row['ali_from'] - 1,
                row['ali_to'],
                row['hmm_name'],
            ))
        return dict(covered)

    def get_uncovered_regions(
        self,
        covered_regions: Dict[str, List[Tuple[int, int, str]]],
        protein_lengths: Dict[str, int],
        protein_ids    : List[str],
        min_length     : int = 20,
        max_length     : int = 300,
        gap_fill       : int = 10,
    ) -> Dict[str, List[Tuple[int, int]]]:
        uncovered = {}
        for pid in protein_ids:
            plen    = protein_lengths.get(pid, 0)
            if plen == 0:
                continue
            regions = covered_regions.get(pid, [])
            merged  = self._merge_intervals(
                [(s, e) for s, e, _ in regions], gap=gap_fill
            )
            uncov = [
                (s, e) for s, e in self._complement(merged, 0, plen)
                if min_length <= (e - s) <= max_length
            ]
            if uncov:
                uncovered[pid] = uncov
        total = sum(len(v) for v in uncovered.values())
        print(f"  Uncovered segments: {total} segments from {len(uncovered)} proteins")
        return uncovered

    def compute_enrichment(
        self,
        filtered_df : pd.DataFrame,
        rbp_ids     : Set[str],
        all_proteins: List[str],
    ) -> pd.DataFrame:
        all_set     = set(all_proteins)
        non_rbp_set = all_set - rbp_ids
        stats       = []
        for hmm_name, grp in filtered_df.groupby('hmm_name'):
            hit_pids = set(grp['protein_id'].unique())
            rp_w     = len(hit_pids & rbp_ids)
            rp_wo    = len(rbp_ids) - rp_w
            nr_w     = len(hit_pids & non_rbp_set)
            nr_wo    = len(non_rbp_set) - nr_w
            try:
                _, pval = fisher_exact(
                    [[rp_w, rp_wo], [nr_w, nr_wo]], alternative='greater'
                )
            except Exception:
                pval = 1.0
            rp_freq = rp_w / max(len(rbp_ids), 1)
            nr_freq = nr_w / max(len(non_rbp_set), 1)
            stats.append({
                'hmm_name'         : hmm_name,
                'rbp_hits'         : rp_w,
                'non_rbp_hits'     : nr_w,
                'rbp_frequency'    : rp_freq,
                'non_rbp_frequency': nr_freq,
                'enrichment_fold'  : rp_freq / (nr_freq + 1e-9),
                'p_value'          : pval,
                'neg_log10_p'      : -np.log10(max(pval, 1e-20)),
                'avg_score'        : grp['domain_score'].mean(),
            })
        df  = pd.DataFrame(stats).sort_values('enrichment_fold', ascending=False)
        sig = ((df['p_value'] < 0.01) & (df['enrichment_fold'] >= 2.0)).sum()
        print(f"  Significantly enriched HMMs: {sig} / {len(df)}")
        return df

    @staticmethod
    def _merge_intervals(
        intervals: List[Tuple[int, int]], gap: int = 0
    ) -> List[Tuple[int, int]]:
        if not intervals:
            return []
        intervals = sorted(intervals)
        merged    = [list(intervals[0])]
        for s, e in intervals[1:]:
            if s <= merged[-1][1] + gap:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        return [(s, e) for s, e in merged]

    @staticmethod
    def _complement(
        intervals: List[Tuple[int, int]], lo: int, hi: int
    ) -> List[Tuple[int, int]]:
        if not intervals:
            return [(lo, hi)]
        result, cur = [], lo
        for s, e in sorted(intervals):
            if cur < s:
                result.append((cur, s))
            cur = max(cur, e)
        if cur < hi:
            result.append((cur, hi))
        return result




class ESMTokenAttribution:
    def __init__(self, device: torch.device, cache_dir: str = './token_attr_cache'):
        self.device    = device
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._model = self._alphabet = self._bc = None

    def _load(self):
        if self._model is None:
            print("  Loading ESM-2 650M (token attribution)...")
            self._model, self._alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            self._bc    = self._alphabet.get_batch_converter()
            self._model = self._model.to(self.device).eval()

    def _unload(self):
        del self._model, self._alphabet, self._bc
        self._model = self._alphabet = self._bc = None
        clear_memory()

    @torch.no_grad()
    def compute_batch(
        self,
        protein_ids: List[str],
        sequences  : Dict[str, str],
        batch_size : int = 4,
    ) -> Dict[str, np.ndarray]:
        cache_file = os.path.join(self.cache_dir, 'token_importance.pkl')
        result: Dict[str, np.ndarray] = {}

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                result = pickle.load(f)
            hit = sum(1 for p in protein_ids if p in result)
            print(f"  Token attribution cache hit: {hit}/{len(protein_ids)}")

        to_compute = [p for p in protein_ids if p not in result]

        if to_compute:
            self._load()
            for i in tqdm(range(0, len(to_compute), batch_size),
                          desc="  ESM token attribution"):
                batch_pids = to_compute[i:i + batch_size]
                batch_data, clean = [], {}
                for pid in batch_pids:
                    seq = ''.join(
                        aa for aa in sequences.get(pid, 'A').upper()
                        if aa in 'ACDEFGHIKLMNPQRSTVWY'
                    )
                    seq = seq[:1022] if len(seq) > 1022 else (seq or 'A')
                    clean[pid] = seq
                    batch_data.append((pid, seq))
                try:
                    _, batch_strs, tokens = self._bc(batch_data)
                    tokens = tokens.to(self.device)
                    out    = self._model(
                        tokens, repr_layers=[33], return_contacts=False
                    )
                    reps   = out["representations"][33]
                    for j, pid in enumerate(batch_pids):
                        slen  = len(batch_strs[j])
                        vec   = reps[j, 1:slen + 1].cpu().float()
                        norms = vec.norm(dim=-1).numpy()
                        lo, hi = norms.min(), norms.max()
                        result[pid] = (norms - lo) / (hi - lo + 1e-8)
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                        for pid in batch_pids:
                            result[pid] = np.zeros(len(clean.get(pid, 'A')))
                    else:
                        raise
                if i % 10 == 0:
                    clear_memory()

            self._unload()
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)

        return result




class NovelDomainExtractor:
    AA_ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'

    def __init__(
        self,
        sequences  : Dict[str, str],
        labels     : Dict[str, int],
        predictions: Dict[str, float],
    ):
        self.sequences   = sequences
        self.labels      = labels
        self.predictions = predictions

    def extract_candidate_segments(
        self,
        uncovered_regions: Dict[str, List[Tuple[int, int]]],
        token_importance : Dict[str, np.ndarray],
        min_rbp_score    : float = 0.55,
    ) -> pd.DataFrame:
        high_conf = {
            pid for pid, sc in self.predictions.items()
            if sc >= min_rbp_score and pid in uncovered_regions
        }
        print(f"  High-confidence candidate proteins: {len(high_conf)}")
        records = []
        for pid in tqdm(high_conf, desc="  Extracting candidate segments"):
            seq        = self.sequences.get(pid, '')
            importance = token_importance.get(pid, np.zeros(len(seq)))
            for start, end in uncovered_regions[pid]:
                region_seq = seq[start:end]
                if not region_seq:
                    continue
                eff_s = min(start, len(importance))
                eff_e = min(end, len(importance))
                if eff_s < eff_e:
                    sl       = importance[eff_s:eff_e]
                    imp_mean = float(sl.mean())
                    imp_max  = float(sl.max())
                else:
                    imp_mean = imp_max = 0.0
                records.append({
                    'protein_id'     : pid,
                    'start'          : start,
                    'end'            : end,
                    'length'         : end - start,
                    'sequence'       : region_seq,
                    'rbp_score'      : self.predictions[pid],
                    'importance_mean': imp_mean,
                    'importance_max' : imp_max,
                    'is_rbp'         : self.labels.get(pid, 0),
                    'combined_score' : self.predictions[pid] * imp_mean,
                })
        df = (pd.DataFrame(records)
              .sort_values('combined_score', ascending=False)
              .reset_index(drop=True))
        print(f"  Total candidate segments: {len(df)}")
        return df

    def cluster(
        self,
        candidates_df  : pd.DataFrame,
        min_importance : float = 0.20,
        eps            : float = 0.40,
        min_samples    : int   = 5,
    ) -> List[Dict]:
        df = candidates_df[
            candidates_df['importance_mean'] >= min_importance
        ].copy().reset_index(drop=True)
        print(f"  Candidates after filtering: {len(df)}")
        if len(df) < min_samples:
            print("  Insufficient candidate segments, skipping clustering")
            return []

        vectors = self._featurize(df['sequence'].tolist())
        vec_std = StandardScaler().fit_transform(vectors)
        n_comp  = min(20, vec_std.shape[0] - 1, vec_std.shape[1])
        vec_pca = PCA(n_components=n_comp, random_state=42).fit_transform(vec_std)
        labels  = DBSCAN(
            eps=eps, min_samples=min_samples, metric='cosine', n_jobs=-1
        ).fit_predict(vec_pca)

        n_clust = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"  Clusters: {n_clust}, noise points: {(labels == -1).sum()}")

        total_rbp = max(sum(self.labels.values()), 1)
        total_all = max(len(self.labels), 1)
        results   = []

        for cid in sorted(set(labels)):
            if cid == -1:
                continue
            mask    = labels == cid
            members = df[mask].to_dict('records')
            pids_c  = list(set(r['protein_id'] for r in members))
            seqs_c  = [r['sequence'] for r in members]

            rbp_in   = sum(1 for r in members if r['is_rbp'] == 1)
            total_in = len(members)
            rbp_out  = total_rbp - rbp_in
            nr_in    = total_in - rbp_in
            nr_out   = (total_all - total_rbp) - nr_in
            try:
                _, pval = fisher_exact(
                    [[rbp_in, rbp_out], [nr_in, nr_out]], alternative='greater'
                )
            except Exception:
                pval = 1.0

            rbp_f = rbp_in / total_in if total_in > 0 else 0.0
            bg_f  = total_rbp / total_all
            fold  = rbp_f / (bg_f + 1e-9)
            cons  = self._conservation(seqs_c)
            aa_c  = self._aa_composition(seqs_c)

            results.append({
                'cluster_id'       : int(cid),
                'size'             : int(mask.sum()),
                'protein_count'    : len(pids_c),
                'proteins'         : pids_c[:30],
                'sequences'        : seqs_c[:10],
                'consensus'        : self._consensus(seqs_c),
                'conservation'     : float(cons),
                'aa_composition'   : aa_c,
                'characteristic_aa': sorted(aa_c, key=aa_c.get, reverse=True)[:5],
                'rbp_enrichment'   : float(fold),
                'rbp_pvalue'       : float(pval),
                'neg_log10_p'      : float(-np.log10(max(pval, 1e-20))),
                'avg_length'       : float(np.mean([r['length'] for r in members])),
                'avg_importance'   : float(np.mean([r['importance_mean'] for r in members])),
                'avg_rbp_score'    : float(np.mean([r['rbp_score'] for r in members])),
                'combined_score'   : float(
                    fold * cons * np.mean([r['importance_mean'] for r in members])
                ),
                'is_significant'   : pval < 0.01 and fold >= 2.0 and len(pids_c) >= 8,
            })

        results.sort(key=lambda x: x['combined_score'], reverse=True)
        print(f"  Significant novel domain candidates: {sum(1 for c in results if c['is_significant'])}")
        return results

    def _featurize(self, seqs: List[str]) -> np.ndarray:
        vecs = []
        for seq in seqs:
            seq = seq.upper()
            n   = max(len(seq), 1)
            aa_freq = np.zeros(20)
            for aa in seq:
                if aa in self.AA_ALPHABET:
                    aa_freq[self.AA_ALPHABET.index(aa)] += 1
            aa_freq /= n
            def frac(chars):
                return sum(1 for a in seq if a in chars) / n
            phys = np.array([
                frac('RKH'), frac('DE'), frac('FYWH'), frac('AVILM'),
                frac('STNQ'), frac('GP'),
                frac('RKH') - frac('DE'),
                seq.count('RGG') / n, seq.count('GGG') / n,
                seq.count('RG')  / n, seq.count('GR')  / n,
                seq.count('KH')  / n,
                len(set(seq) & set(self.AA_ALPHABET)) / 20,
                np.log1p(len(seq)) / 10,
                frac('FY'),
            ])
            vecs.append(np.concatenate([aa_freq, phys]))
        return np.array(vecs, dtype=np.float32)

    def _consensus(self, seqs: List[str]) -> str:
        if not seqs:
            return ""
        tlen    = int(np.median([len(s) for s in seqs]))
        aligned = [
            s[:tlen] if len(s) >= tlen else s + '-' * (tlen - len(s))
            for s in seqs
        ]
        result = []
        for pos in range(tlen):
            cnt = Counter(
                s[pos] for s in aligned
                if pos < len(s) and s[pos] in self.AA_ALPHABET
            )
            if not cnt:
                result.append('-')
                continue
            top, top_c = cnt.most_common(1)[0]
            freq = top_c / len(seqs)
            result.append(
                top       if freq > 0.70 else
                top.lower() if freq > 0.40 else 'x'
            )
        return ''.join(result)

    def _conservation(self, seqs: List[str]) -> float:
        if len(seqs) < 2:
            return 1.0
        tlen = int(np.median([len(s) for s in seqs]))
        ents = []
        for pos in range(tlen):
            cnt = Counter(
                s[pos] for s in seqs
                if pos < len(s) and s[pos] in self.AA_ALPHABET
            )
            if not cnt:
                continue
            tot = sum(cnt.values())
            ent = -sum((c / tot) * np.log2(c / tot) for c in cnt.values() if c > 0)
            ents.append(ent)
        return float(1.0 - np.mean(ents) / np.log2(20)) if ents else 0.0

    def _aa_composition(self, seqs: List[str]) -> Dict[str, float]:
        cnt, total = Counter(), 0
        for seq in seqs:
            for aa in seq.upper():
                if aa in self.AA_ALPHABET:
                    cnt[aa] += 1
                    total   += 1
        return {aa: cnt[aa] / max(total, 1) for aa in self.AA_ALPHABET}




class NovelHMMBuilder:
    def __init__(self, output_dir: str):
        self.output_dir      = output_dir
        self.hmmer_available = self._check('hmmbuild')
        self.mafft_available = self._check('mafft')
        os.makedirs(output_dir, exist_ok=True)

    @staticmethod
    def _check(tool: str) -> bool:
        try:
            subprocess.run([tool, '--version'], capture_output=True, timeout=5)
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print(f"  WARNING: {tool} not found, related steps will be skipped")
            return False

    def build_for_cluster(
        self, cluster: Dict, idx: int, min_seqs: int = 5
    ) -> Optional[str]:
        if len(cluster['sequences']) < min_seqs or not self.hmmer_available:
            return None
        cdir     = os.path.join(self.output_dir, f'cluster_{idx:03d}')
        os.makedirs(cdir, exist_ok=True)
        fasta_p  = os.path.join(cdir, 'seqs.fasta')
        aln_p    = os.path.join(cdir, 'seqs.aln')
        sto_p    = os.path.join(cdir, 'seqs.sto')
        hmm_p    = os.path.join(cdir, f'cluster_{idx:03d}.hmm')
        hmm_name = 'NOVEL_RBP_{:03d}_{}'.format(
            idx, ''.join(cluster['characteristic_aa'][:5])
        )
        with open(fasta_p, 'w') as f:
            for i, seq in enumerate(cluster['sequences']):
                f.write(f">seq_{i:04d}\n{seq}\n")
        aln_ok = False
        if self.mafft_available:
            try:
                with open(aln_p, 'w') as f:
                    r = subprocess.run(
                        ['mafft', '--auto', '--quiet', '--thread', '4', fasta_p],
                        stdout=f, stderr=subprocess.DEVNULL, timeout=120
                    )
                if r.returncode == 0:
                    self._to_stockholm(aln_p, sto_p)
                    aln_ok = True
            except Exception:
                pass
        inp = sto_p if aln_ok else fasta_p
        try:
            r = subprocess.run(
                ['hmmbuild', '-n', hmm_name, '--cpu', '4', hmm_p, inp],
                capture_output=True, text=True, timeout=180
            )
            if r.returncode == 0:
                print(f"    [OK] cluster_{idx:03d}: {hmm_name}")
                return hmm_p
        except Exception as e:
            print(f"    [FAIL] cluster_{idx:03d}: {e}")
        return None

    def merge_and_press(self, hmm_paths: List[str], output: str) -> str:
        valid = [p for p in hmm_paths if p and os.path.exists(p)]
        if not valid:
            return ""
        with open(output, 'w') as out_f:
            for p in valid:
                with open(p) as in_f:
                    out_f.write(in_f.read())
        if self.hmmer_available:
            subprocess.run(['hmmpress', '-f', output], capture_output=True)
        print(f"  Novel HMM database: {output} ({len(valid)} HMMs)")
        return output

    @staticmethod
    def _to_stockholm(fasta_aln: str, sto_out: str):
        seqs, cid = {}, None
        with open(fasta_aln) as f:
            for line in f:
                line = line.rstrip()
                if line.startswith('>'):
                    cid = line[1:].split()[0]
                    seqs[cid] = ''
                elif cid:
                    seqs[cid] += line
        with open(sto_out, 'w') as f:
            f.write("# STOCKHOLM 1.0\n\n")
            for sid, seq in seqs.items():
                f.write(f"{sid:<30s} {seq}\n")
            f.write("//\n")




def run_domain_mode(args):
    print("\n" + "="*55)
    print("Mode 3: Novel domain identification (5-fold ensemble)")
    print("="*55)
    os.makedirs(args.output_dir, exist_ok=True)

    
    print("\n[Step 1] Loading data...")
    rbp_ids   = load_rbp_ids(args.rbp_list)
    sequences = load_fasta(args.fasta)
    adjacency = load_ppi(args.ppi, sequences, min_score=0.60)

    valid_proteins = [
        p for p in sequences
        if len(adjacency.get(p, [])) >= args.min_degree
    ]
    labels = {p: (1 if p in rbp_ids else 0) for p in valid_proteins}
    pos_n  = sum(labels.values())
    print(f"  Valid proteins: {len(valid_proteins)} | RBP: {pos_n} | non-RBP: {len(labels)-pos_n}")

   
    print("\n[Step 2] Feature extraction...")
    seq_features, seq_dim = prepare_features(
        valid_proteins, sequences,
        cache_dir=args.cache_dir,
        train_cache_path=args.esm_cache,
        model_dir=args.model_dir,
        esm_batch_size=args.esm_batch,
    )

    
    print("\n[Step 3] Building subgraphs...")
    gf     = GraphFeatures(adjacency)
    graphs = {
        p: build_subgraph(p, adjacency, seq_features, gf)
        for p in tqdm(valid_proteins, desc="  Building graphs")
    }
    node_dim = graphs[valid_proteins[0]].x.shape[1]
    print(f"  node_dim={node_dim}, seq_dim={seq_dim}")

    
    print("\n[Step 4] 5-fold ensemble model inference...")
    dataset = PPIDataset(valid_proteins, graphs, seq_features)
    loader  = DataLoader(
        dataset, batch_size=64, shuffle=False,
        collate_fn=collate_ppi, num_workers=0
    )

    fold_preds_list = []

    for fold_i in range(1, args.n_folds + 1):
        model_path = os.path.join(args.model_dir, f'balanced_fold_{fold_i}.pt')
        if not os.path.exists(model_path):
            print(f"  WARNING: fold_{fold_i} not found, skipping")
            continue

        print(f"  Loading Fold {fold_i}...")
        model = BalancedGATModel(
            node_dim=node_dim, seq_dim=seq_dim,
            hidden_dim=384, gat_hidden=128, gat_heads=4,
            num_layers=3, dropout=0.5, use_edge_attr=True
        ).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()

        preds_this_fold = []
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"  Fold {fold_i}", leave=False):
                graph = batch['graph'].to(DEVICE)
                seq   = batch['seq_features'].to(DEVICE)

                if not args.no_tta:
                    ps = [torch.sigmoid(model(graph, seq, training=False)[0].float())]
                    for _ in range(5):
                        noisy = seq + torch.randn_like(seq) * 0.02
                        out, _ = model(graph, noisy, training=False)
                        ps.append(torch.sigmoid(out.float()))
                    p = torch.stack(ps).mean(dim=0)
                else:
                    p = torch.sigmoid(model(graph, seq, training=False)[0].float())

                preds_this_fold.extend(p.cpu().numpy())

        fold_preds_list.append(preds_this_fold)
        clear_memory()

    
    final_preds_arr = np.mean(fold_preds_list, axis=0)
    predictions     = {pid: float(sc) for pid, sc in zip(valid_proteins, final_preds_arr)}

    
    score_tsv = os.path.join(args.output_dir, 'all_protein_scores.tsv')
    pd.DataFrame([
        {
            'protein_id'   : pid,
            'rbp_score'    : sc,
            'is_rbp'       : labels.get(pid, 0),
            'predicted_rbp': int(sc >= 0.5),
        }
        for pid, sc in sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    ]).to_csv(score_tsv, sep='\t', index=False)
    print(f"  Scores saved: {score_tsv}")

    
    print("\n[Step 5] Parsing HMMER results...")
    if not os.path.exists(args.hmmer_result):
        print("\n" + "!"*55)
        print("  HMMER result file not found! Please run first:")
        print(f"  hmmscan --domtblout {args.hmmer_result} \\")
        print(f"    --cpu 16 -E 1e-3 --domE 1e-3 \\")
        print(f"    {args.hmm_db} {args.fasta}")
        print("!"*55)
        return

    parser          = HMMERParser(evalue_thr=1e-5, score_thr=20.0, coverage_thr=0.5)
    raw_df          = parser.parse_domtblout(args.hmmer_result)
    filtered_df     = parser.filter_hits(raw_df)
    prot_lengths    = {p: len(sequences[p]) for p in valid_proteins}
    covered_regions = parser.get_covered_regions(filtered_df)
    uncovered       = parser.get_uncovered_regions(
        covered_regions, prot_lengths, valid_proteins,
        min_length=20, max_length=300, gap_fill=10
    )
    enrich_df = parser.compute_enrichment(filtered_df, rbp_ids, valid_proteins)
    enrich_df.to_csv(
        os.path.join(args.output_dir, 'pfam_enrichment.tsv'),
        sep='\t', index=False
    )

    
    print("\n[Step 6] ESM token attribution...")
    attr_targets = [
        pid for pid in valid_proteins
        if predictions.get(pid, 0) >= 0.55 and pid in uncovered
    ]
    print(f"  Attribution target proteins: {len(attr_targets)}")
    esm_attr = ESMTokenAttribution(
        DEVICE,
        cache_dir=os.path.join(args.output_dir, 'token_attr')
    )
    token_importance = esm_attr.compute_batch(attr_targets, sequences, batch_size=4)

    
    print("\n[Step 7] Candidate segment extraction and clustering...")
    extractor     = NovelDomainExtractor(sequences, labels, predictions)
    candidates_df = extractor.extract_candidate_segments(
        uncovered, token_importance, min_rbp_score=0.55
    )
    candidates_df.to_csv(
        os.path.join(args.output_dir, 'candidate_segments.tsv'),
        sep='\t', index=False
    )
    cluster_results = extractor.cluster(
        candidates_df, min_importance=0.20, eps=0.40, min_samples=5
    )
    with open(os.path.join(args.output_dir, 'novel_domains.json'), 'w') as f:
        json.dump(cluster_results, f, indent=2, default=str)

    
    print("\n[Step 8] Building novel HMMs...")
    builder      = NovelHMMBuilder(os.path.join(args.output_dir, 'hmms'))
    sig_clusters = [c for c in cluster_results if c['is_significant']]
    hmm_paths    = [
        builder.build_for_cluster(c, i, min_seqs=5)
        for i, c in enumerate(sig_clusters)
    ]
    novel_hmm_db = ""
    if any(hmm_paths):
        novel_hmm_db = builder.merge_and_press(
            hmm_paths,
            os.path.join(args.output_dir, 'novel_domains.hmm')
        )

    
    n_sig = sum(1 for c in cluster_results if c['is_significant'])
    print("\n" + "="*55)
    print("  Novel domain identification complete (5-fold ensemble)")
    print(f"  Proteins analyzed:             {len(valid_proteins)}")
    print(f"  Candidate segments:            {len(candidates_df)}")
    print(f"  Total clusters:                {len(cluster_results)}")
    print(f"  Significant novel domains:     {n_sig}")
    print(f"  Novel HMM database:            {novel_hmm_db or 'not generated'}")
    print(f"  Output directory:              {args.output_dir}/")
    print("="*55)



def parse_args():
    parser = argparse.ArgumentParser(
        description='RBP prediction and novel domain identification system v3.0',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Usage examples:
  # Sequence-only prediction
  python rbp_system.py --mode seq \\
      --input proteins.fasta --model_dir ./models --output results.csv

  # PPI + sequence prediction
  python rbp_system.py --mode ppi \\
      --input proteins.fasta --model_dir ./models \\
      --ppi ppi.txt --background all_proteins.fasta --output results.csv

  # Novel domain identification (5-fold ensemble)
  python rbp_system.py --mode domain \\
      --model_dir ./models \\
      --rbp_list ara_rbp_id.txt \\
      --fasta all_proteins.fasta \\
      --ppi ppi.txt \\
      --hmmer_result ara_vs_rbp.domtblout \\
      --esm_cache ./balanced_cache_v2_650M/esm_features_650M.pkl \\
      --output_dir ./novel_domain_results
        """
    )

    # ── General arguments
    parser.add_argument('--mode', required=True,
                        choices=['seq', 'ppi', 'domain'],
                        help='Running mode: seq / ppi / domain')
    parser.add_argument('--model_dir', default='./models',
                        help='Model directory (containing balanced_fold_1~5.pt)')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of ensemble folds, default 5')
    parser.add_argument('--esm_cache', default=None,
                        help='Training-time ESM feature cache path (.pkl) to speed up feature extraction')
    parser.add_argument('--cache_dir', default='./predict_cache',
                        help='Prediction cache directory, default ./predict_cache')
    parser.add_argument('--esm_batch', type=int, default=6,
                        help='ESM extraction batch size, default 6')
    parser.add_argument('--no_tta', action='store_true',
                        help='Disable TTA (faster inference, slightly lower accuracy)')
    parser.add_argument('--seed', type=int, default=42)

    # ── seq / ppi mode arguments
    parser.add_argument('--input', default=None,
                        help='[seq/ppi] Input FASTA file of proteins to predict')
    parser.add_argument('--output', default='rbp_predictions.csv',
                        help='[seq/ppi] Output CSV file path')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='[seq/ppi] RBP classification threshold, default 0.5')
    parser.add_argument('--ppi', default=None,
                        help='[ppi/domain] PPI network file')
    parser.add_argument('--background', default=None,
                        help='[ppi] Background proteome FASTA')

    # ── domain mode arguments
    parser.add_argument('--rbp_list', default=None,
                        help='[domain] Known RBP ID list file')
    parser.add_argument('--fasta', default=None,
                        help='[domain] Full proteome FASTA file')
    parser.add_argument('--hmmer_result', default=None,
                        help='[domain] hmmscan --domtblout output file')
    parser.add_argument('--hmm_db', default=None,
                        help='[domain] Pfam HMM database (used in command hint)')
    parser.add_argument('--output_dir', default='./novel_domain_results',
                        help='[domain] Results output directory')
    parser.add_argument('--min_degree', type=int, default=2,
                        help='[domain] Minimum PPI degree filter, default 2')

    return parser.parse_args()


def validate_args(args):
    if args.mode == 'seq':
        if not args.input:
            raise ValueError("seq mode requires --input")

    elif args.mode == 'ppi':
        missing = [
            x for x, v in [
                ('--input', args.input),
                ('--ppi', args.ppi),
                ('--background', args.background),
            ] if not v
        ]
        if missing:
            raise ValueError(f"ppi mode missing arguments: {', '.join(missing)}")

    elif args.mode == 'domain':
        missing = [
            x for x, v in [
                ('--rbp_list', args.rbp_list),
                ('--fasta', args.fasta),
                ('--ppi', args.ppi),
                ('--hmmer_result', args.hmmer_result),
            ] if not v
        ]
        if missing:
            raise ValueError(f"domain mode missing arguments: {', '.join(missing)}")


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.cache_dir, exist_ok=True)

    print("=" * 55)
    print("  RBP Prediction and Novel Domain Identification System v3.0")
    print(f"  Mode: {args.mode} | Device: {DEVICE}")
    print("=" * 55)

    try:
        validate_args(args)
    except ValueError as e:
        print(f"\n[ERROR] Argument error: {e}")
        sys.exit(1)

    use_tta = not args.no_tta

    if args.mode == 'seq':
        query_seqs = load_fasta(args.input)
        query_ids  = list(query_seqs.keys())
        print(f"\nProteins to predict: {len(query_ids)}")

        result_df = predict_seq_mode(
            query_ids        = query_ids,
            sequences_dict   = query_seqs,
            model_dir        = args.model_dir,
            n_folds          = args.n_folds,
            use_tta          = use_tta,
            threshold        = args.threshold,
            cache_dir        = args.cache_dir,
            train_cache_path = args.esm_cache,
            esm_batch_size   = args.esm_batch,
        )
        result_df.to_csv(args.output, index=False)
        print_summary(result_df, args.output, args.threshold)

    elif args.mode == 'ppi':
        query_seqs = load_fasta(args.input)
        query_ids  = list(query_seqs.keys())
        print(f"\nProteins to predict: {len(query_ids)}")

        result_df = predict_ppi_mode(
            query_ids        = query_ids,
            sequences_dict   = query_seqs,
            ppi_file         = args.ppi,
            background_fasta = args.background,
            model_dir        = args.model_dir,
            n_folds          = args.n_folds,
            use_tta          = use_tta,
            threshold        = args.threshold,
            cache_dir        = args.cache_dir,
            train_cache_path = args.esm_cache,
            esm_batch_size   = args.esm_batch,
        )
        result_df.to_csv(args.output, index=False)
        print_summary(result_df, args.output, args.threshold)

    elif args.mode == 'domain':
        run_domain_mode(args)


if __name__ == '__main__':
    main()
