#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Balanced PPI-GAT RBP Prediction Model v2.4 (The Final Push)

"""

import os
import gc
import json
import pickle
import warnings
import random
import math
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
from copy import deepcopy

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from torch.optim.swa_utils import AveragedModel, SWALR

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, roc_curve,
    average_precision_score, matthews_corrcoef
)
from sklearn.preprocessing import RobustScaler
from tqdm.auto import tqdm

from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data, Batch
from torch_geometric.utils import softmax

try:
    import esm
except ImportError:
    raise ImportError("请先安装 esm: pip install esm")

warnings.filterwarnings('ignore')


class BalancedConfig:
    
    RBP_FILE = "ara_rbp_id.txt"
    FASTA_FILE = "Arabidopsis_thaliana.TAIR10.pep.all.fasta"
    PPI_FILE = "3702.protein.v12.0.Arabidopsis_thaliana.TAIR10.pep.all_1.txt"
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
   
    TEST_SIZE = 0.15
    N_FOLDS = 5
    MIN_PPI_SCORE = 0.60
    MIN_DEGREE = 2
    
  
    MAX_NEIGHBORS = 40  
    USE_EDGE_WEIGHT = True
    

    HIDDEN_DIM = 384     
    GAT_HIDDEN = 128
    GAT_HEADS = 4
    GAT_LAYERS = 3
    DROPOUT = 0.5
    USE_EDGE_ATTR = True
    AUX_LOSS_WEIGHT = 0.2 
    
    
    EPOCHS = 60
    LEARNING_RATE = 6e-4 
    MIN_LR = 1e-6
    WEIGHT_DECAY = 0.05
    WARMUP_EPOCHS = 5
    PATIENCE = 16
    
    # SWA
    USE_SWA = True
    SWA_START = 15       
    SWA_LR = 2e-4
    
  
    FOCAL_ALPHA = 0.60
    FOCAL_GAMMA = 2.5
    LABEL_SMOOTHING = 0.1
    
   
    ESM_BATCH_SIZE = 6
    TRAIN_BATCH_SIZE = 48
    VAL_BATCH_SIZE = 96
    
    
    USE_MIXUP = True
    MIXUP_ALPHA = 0.6
    USE_FEATURE_DROPOUT = True
    FEATURE_DROPOUT = 0.15
    
   
    USE_WEIGHTED_ENSEMBLE = True
    USE_TTA = True
    
    USE_AMP = True
    GRADIENT_CLIP = 1.0
    
    CACHE_DIR = './balanced_cache_v2_650M'
    USE_CACHE = True
    SEED = 42

config = BalancedConfig()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

set_seed(config.SEED)
os.makedirs(config.CACHE_DIR, exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('models', exist_ok=True)

print(f"Device: {config.DEVICE}")

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class DataProcessor:
    @staticmethod
    def load_rbp_ids(filepath: str) -> set:
        with open(filepath, 'r') as f:
            return set(line.strip() for line in f if line.strip())
    
    @staticmethod
    def load_sequences(filepath: str) -> Dict[str, str]:
        sequences = {}
        current_id, current_seq = None, []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_id and current_seq:
                        sequences[current_id] = ''.join(current_seq)
                    current_id = line[1:].split()[0]
                    current_seq = []
                else:
                    current_seq.append(line)
            if current_id and current_seq:
                sequences[current_id] = ''.join(current_seq)
        return sequences
    
    @staticmethod
    def load_ppi(filepath: str, sequences: Dict, min_score: float = 0.7) -> Dict[str, List[Tuple[str, float]]]:
        adjacency = defaultdict(list)
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    p1, p2 = parts[0], parts[1]
                    score = float(parts[2]) if len(parts) > 2 else 1.0
                    if p1 in sequences and p2 in sequences and score >= min_score and p1 != p2:
                        adjacency[p1].append((p2, score))
                        adjacency[p2].append((p1, score))
        for k in adjacency:
            seen = {}
            for neighbor, score in adjacency[k]:
                if neighbor not in seen or seen[neighbor] < score:
                    seen[neighbor] = score
            adjacency[k] = sorted(seen.items(), key=lambda x: x[1], reverse=True)
        return dict(adjacency)


class BalancedFeatureExtractor:
    def __init__(self):
        self.amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        self.aa_to_idx = {aa: i for i, aa in enumerate(self.amino_acids)}
        self.rbp_motifs = ['RGG', 'GGG', 'RRM', 'KH', 'SR', 'RS', 'RR', 'KK', 'GG', 'RGR', 'KGK', 'YGG', 'FGG', 'DEAD', 'DEAH', 'GGGG', 'RGGR', 'GAR', 'RG', 'GR', 'KG', 'GK']
    
    def extract(self, sequence: str) -> np.ndarray:
        seq = ''.join([aa for aa in sequence.upper() if aa in self.amino_acids])
        if len(seq) < 5: seq = seq + 'A' * (5 - len(seq))
        features = []
        
        
        aa_comp = np.zeros(20)
        for aa in seq: aa_comp[self.aa_to_idx[aa]] += 1
        features.extend(aa_comp / len(seq))
        
        for motif in self.rbp_motifs: features.append(seq.count(motif) / len(seq))
        
        features.append(sum(1 for aa in seq if aa in 'RKH') / len(seq))
        features.append(sum(1 for aa in seq if aa in 'DE') / len(seq))
        features.append(len(seq) / 1000)
        features.append(np.log1p(len(seq)) / 10)
        
        charges = [1 if aa in 'RKH' else (-1 if aa in 'DE' else 0) for aa in seq]
        features.append(np.std(charges))
        
        return np.array(features, dtype=np.float32)

class BalancedGraphFeatures:
    def __init__(self, adjacency: Dict[str, List[Tuple[str, float]]]):
        self.adjacency = adjacency
        self.simple_adj = {k: [n for n, _ in v] for k, v in adjacency.items()}
        self.degrees = {n: len(neighbors) for n, neighbors in self.simple_adj.items()}
        self.max_degree = max(self.degrees.values()) if self.degrees else 1
    
    def extract(self, node: str) -> np.ndarray:
        degree = self.degrees.get(node, 0)
        features = [degree/100.0, np.log1p(degree)/5.0, degree/self.max_degree]
        
        if node in self.adjacency and self.adjacency[node]:
            weights = [s for _, s in self.adjacency[node]]
            features.extend([np.mean(weights), np.max(weights)])
        else:
            features.extend([0.0, 0.0])
        return np.array(features, dtype=np.float32)

class BalancedESMExtractor:
    def __init__(self, device: torch.device):
        self.device = device
        self.model = None
        self.batch_converter = None
        self.alphabet = None
        self.repr_layer = 33
    
    def _load_model(self):
        if self.model is None:
            print("Loading ESM-2 650M model...")
            self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            self.batch_converter = self.alphabet.get_batch_converter()
            self.model = self.model.to(self.device).eval()
            self.hidden_dim = self.model.embed_dim
    
    def extract(self, protein_ids: List[str], sequences_dict: Dict[str, str], batch_size: int = 6) -> Dict[str, torch.Tensor]:
        cache_file = os.path.join(config.CACHE_DIR, 'esm_features_650M.pkl')
        if config.USE_CACHE and os.path.exists(cache_file):
            print("Loading cached ESM features...")
            with open(cache_file, 'rb') as f:
                cached = pickle.load(f)
                if all(pid in cached for pid in protein_ids): return cached
        else: cached = {}
        
        self._load_model()
        features = cached.copy()
        to_compute = [pid for pid in protein_ids if pid not in features]
        
        if to_compute:
            print(f"Computing features for {len(to_compute)} proteins...")
            for i in tqdm(range(0, len(to_compute), batch_size), desc="ESM-650M"):
                batch_pids = to_compute[i:i+batch_size]
                batch_data = []
                for pid in batch_pids:
                    raw_seq = sequences_dict.get(pid, 'A')
                    seq = ''.join([aa for aa in raw_seq.upper() if aa in 'ACDEFGHIKLMNPQRSTVWY'])
                    if len(seq) > 1022: seq = seq[:1022]
                    elif len(seq) == 0: seq = 'A'
                    batch_data.append((pid, seq))
                
                try:
                    batch_labels, batch_strs, batch_tokens = self.batch_converter(batch_data)
                    batch_tokens = batch_tokens.to(self.device)
                    with torch.no_grad():
                        results = self.model(batch_tokens, repr_layers=[self.repr_layer], return_contacts=False)
                        token_representations = results["representations"][self.repr_layer]
                    
                    for j, pid in enumerate(batch_pids):
                        seq_len = len(batch_strs[j])
                        seq_feat = token_representations[j, 1 : seq_len + 1]
                        # 3种Pooling: Mean, Max, Weighted
                        mean_pool = seq_feat.mean(dim=0)
                        max_pool = seq_feat.max(dim=0)[0]
                        norms = seq_feat.norm(dim=-1, keepdim=True)
                        weights = F.softmax(norms.squeeze(-1), dim=0).unsqueeze(-1)
                        weighted_pool = (seq_feat * weights).sum(dim=0)
                        features[pid] = torch.cat([mean_pool, max_pool, weighted_pool], dim=-1).cpu()
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                        for pid in batch_pids: features[pid] = torch.zeros(self.hidden_dim * 3)
                    else: raise e
                if i % 20 == 0: clear_memory()
            
            if config.USE_CACHE:
                with open(cache_file, 'wb') as f: pickle.dump(features, f)
        
        del self.model
        del self.batch_converter
        del self.alphabet
        self.model = None
        clear_memory()
        return features


def build_balanced_subgraph(target, adjacency, node_features, graph_features, max_neighbors):
    neighbors_with_scores = adjacency.get(target, [])[:max_neighbors]
    neighbors = [n for n, _ in neighbors_with_scores]
    neighbor_weights = {n: s for n, s in neighbors_with_scores}
    nodes = [target] + neighbors
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    
    feat_list = []
    for i, node in enumerate(nodes):
        base_feat = node_features.get(node, node_features[target])
        graph_feat = graph_features.extract(node)
        is_target = 1.0 if i == 0 else 0.0
        edge_weight = neighbor_weights.get(node, 0.5)
        pos_feat = torch.tensor([is_target, edge_weight], dtype=torch.float32)
        combined = torch.cat([base_feat, torch.tensor(graph_feat), pos_feat])
        feat_list.append(combined)
    
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
    edge_attr = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(-1)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(nodes))


class EdgeWeightedGATConv(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4, dropout=0.3, concat=True):
        super().__init__()
        self.gat = GATv2Conv(in_dim, out_dim, heads=heads, dropout=dropout, concat=concat, edge_dim=1)
        out_features = out_dim * heads if concat else out_dim
        self.norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, edge_index, edge_attr=None):
        out = self.gat(x, edge_index, edge_attr=edge_attr)
        out = self.norm(out)
        return F.elu(self.dropout(out))

class MultiScalePooling(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.att = nn.Sequential(nn.Linear(in_dim, in_dim // 4), nn.Tanh(), nn.Linear(in_dim // 4, 1))
        self.gate = nn.Sequential(nn.Linear(in_dim * 3, in_dim), nn.Sigmoid())
    def forward(self, x, batch):
        x = x.float()
        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        scores = self.att(x).squeeze(-1)
        weights = softmax(scores, batch)
        att_pool = global_add_pool(x * weights.unsqueeze(-1), batch)
        concat = torch.cat([mean_pool, max_pool, att_pool], dim=-1)
        gate = self.gate(concat)
        fused = gate * mean_pool + (1 - gate) * (max_pool + att_pool) / 2
        return torch.cat([fused, att_pool], dim=-1)

class BalancedGATModel(nn.Module):
    def __init__(self, node_dim: int, seq_dim: int, hidden_dim: int = 384,
                 gat_hidden: int = 128, gat_heads: int = 4, num_layers: int = 3,
                 dropout: float = 0.5, use_edge_attr: bool = True):
        super().__init__()
        
        
        self.node_proj = nn.Sequential(
            nn.Linear(node_dim, gat_hidden * gat_heads),
            nn.LayerNorm(gat_hidden * gat_heads),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.gat_layers = nn.ModuleList()
        current_dim = gat_hidden * gat_heads
        for i in range(num_layers):
            concat = (i != num_layers - 1)
            self.gat_layers.append(EdgeWeightedGATConv(current_dim, gat_hidden, heads=gat_heads, dropout=dropout, concat=concat))
            current_dim = gat_hidden * gat_heads if concat else gat_hidden
        
        self.pooling = MultiScalePooling(current_dim)
        pool_out_dim = current_dim * 2
        
        self.graph_encoder = nn.Sequential(
            nn.Linear(pool_out_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
       
        self.aux_classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1)
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
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
       
        fusion_dim = hidden_dim // 2 + hidden_dim // 2
        self.cross_attention = nn.MultiheadAttention(hidden_dim // 2, num_heads=4, dropout=dropout, batch_first=True)
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim + hidden_dim // 2, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.feature_dropout = nn.Dropout(config.FEATURE_DROPOUT)
        self.use_edge_attr = use_edge_attr
    
    def forward(self, graph_data, seq_features, training=True):
        x = graph_data.x
        edge_index = graph_data.edge_index
        edge_attr = graph_data.edge_attr if self.use_edge_attr else None
        batch = graph_data.batch
        
        if training and self.training:
            seq_features = self.feature_dropout(seq_features)
        
        # Graph
        x = self.node_proj(x)
        for gat_layer in self.gat_layers: x = gat_layer(x, edge_index, edge_attr)
        graph_pool = self.pooling(x, batch)
        graph_out = self.graph_encoder(graph_pool)
        
        
        aux_out = self.aux_classifier(graph_out).squeeze(-1)
        
        # Seq
        seq_out = self.seq_encoder(seq_features.float())
        
        # Fusion
        graph_q = graph_out.unsqueeze(1)
        seq_kv = seq_out.unsqueeze(1)
        cross_out, _ = self.cross_attention(graph_q, seq_kv, seq_kv)
        cross_out = cross_out.squeeze(1)
        
        combined = torch.cat([graph_out, seq_out, cross_out], dim=-1)
        main_out = self.classifier(combined).squeeze(-1)
        
        return main_out, aux_out


class BalancedTrainer:
    def __init__(self, model, device, criterion, optimizer, use_amp=True, gradient_clip=1.0, use_mixup=True, mixup_alpha=0.4):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.use_amp = use_amp
        self.scaler = GradScaler() if use_amp else None
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.gradient_clip = gradient_clip
    
    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        all_preds, all_labels = [], []
        
        for batch in loader:
            graph = batch['graph'].to(self.device)
            seq_feat = batch['seq_features'].to(self.device)
            targets = batch['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            
            
            if self.use_mixup and np.random.random() < 0.5:
                seq_feat, targets_a, targets_b, lam = mixup_data(seq_feat, targets, self.mixup_alpha)
            else:
                targets_a, targets_b, lam = targets, targets, 1.0
            
            if self.use_amp:
                with autocast():
                    main_out, aux_out = self.model(graph, seq_feat, training=True)
                    # Main Loss
                    loss_main = lam * self.criterion(main_out, targets_a) + (1 - lam) * self.criterion(main_out, targets_b)
                    # Aux Loss (No Mixup usually helps aux task stay grounded)
                    loss_aux = self.criterion(aux_out, targets) 
                    
                    loss = loss_main + config.AUX_LOSS_WEIGHT * loss_aux
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                main_out, aux_out = self.model(graph, seq_feat, training=True)
                loss_main = lam * self.criterion(main_out, targets_a) + (1 - lam) * self.criterion(main_out, targets_b)
                loss_aux = self.criterion(aux_out, targets)
                loss = loss_main + config.AUX_LOSS_WEIGHT * loss_aux
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.optimizer.step()
            
            total_loss += loss.item()
            with torch.no_grad():
                all_preds.extend(torch.sigmoid(main_out.float()).cpu().numpy())
                all_labels.extend(batch['labels'].numpy())
        
        auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5
        return total_loss / len(loader), auc
    
    @torch.no_grad()
    def evaluate(self, loader, use_tta=False):
        self.model.eval()
        all_preds, all_labels = [], []
        
        for batch in loader:
            graph = batch['graph'].to(self.device)
            seq_feat = batch['seq_features'].to(self.device)
            targets = batch['labels'].to(self.device)
            
            main_out, _ = self.model(graph, seq_feat, training=False)
            
            if use_tta:
                tta_preds = [torch.sigmoid(main_out)]
               
                for _ in range(8):
                    noisy_seq = seq_feat + torch.randn_like(seq_feat) * 0.02
                    out, _ = self.model(graph, noisy_seq, training=False)
                    tta_preds.append(torch.sigmoid(out))
                preds = torch.stack(tta_preds).mean(dim=0)
            else:
                preds = torch.sigmoid(main_out)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
        
        preds = np.array(all_preds)
        labels = np.array(all_labels)
        fpr, tpr, thresholds = roc_curve(labels, preds)
        opt_thresh = thresholds[np.argmax(tpr - fpr)]
        preds_binary = (preds > 0.5).astype(int)
        
        return {
            'auc': roc_auc_score(labels, preds),
            'f1': f1_score(labels, preds_binary, zero_division=0),
            'ap': average_precision_score(labels, preds)
        }


class FocalLossWithSmoothing(nn.Module):
    def __init__(self, alpha=0.6, gamma=2.5, smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing
    def forward(self, inputs, targets):
        inputs = inputs.float()
        targets = targets.float()
        if self.smoothing > 0: targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        return ( (self.alpha * targets + (1 - self.alpha) * (1 - targets)) * (1 - pt) ** self.gamma * bce ).mean()

class RBPDataset(Dataset):
    def __init__(self, protein_ids, graphs, seq_features, labels):
        self.protein_ids = protein_ids
        self.graphs = graphs
        self.seq_features = seq_features
        self.labels = labels
    def __len__(self): return len(self.protein_ids)
    def __getitem__(self, idx):
        pid = self.protein_ids[idx]
        return {'protein_id': pid, 'graph': self.graphs[pid], 'seq_features': self.seq_features[pid], 'label': float(self.labels[pid])}

def collate_fn(batch):
    return {'graph': Batch.from_data_list([item['graph'] for item in batch]),
            'seq_features': torch.stack([item['seq_features'] for item in batch]),
            'labels': torch.tensor([item['label'] for item in batch], dtype=torch.float32),
            'protein_ids': [item['protein_id'] for item in batch]}

def mixup_data(seq_features, labels, alpha=0.4):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    lam = max(lam, 1 - lam)
    idx = torch.randperm(seq_features.size(0)).to(seq_features.device)
    return lam * seq_features + (1 - lam) * seq_features[idx], labels, labels[idx], lam

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
    def step(self, epoch):
        if epoch < self.warmup_epochs: lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else: lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * (epoch - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)))
        for pg in self.optimizer.param_groups: pg['lr'] = lr
        return lr


def main():
    print("=" * 80)
    print("Balanced PPI-GAT RBP Prediction Model v2.4 (Aux Loss + Progressive Compression)")
    print("=" * 80)
    
    # 1. Load Data
    rbp_ids = DataProcessor.load_rbp_ids(config.RBP_FILE)
    sequences = DataProcessor.load_sequences(config.FASTA_FILE)
    adjacency = DataProcessor.load_ppi(config.PPI_FILE, sequences, config.MIN_PPI_SCORE)
    
    # 2. Filter
    valid_proteins = [p for p in sequences if len(adjacency.get(p, [])) >= config.MIN_DEGREE]
    labels = {p: (1 if p in rbp_ids else 0) for p in valid_proteins}
    pos_count = sum(labels.values())
    neg_count = len(labels) - pos_count
    print(f"Stats: Pos={pos_count}, Neg={neg_count}, Ratio={neg_count/pos_count:.1f}")
    
    # 3. Split
    train_val_proteins, test_proteins = train_test_split(
        valid_proteins, test_size=config.TEST_SIZE, 
        stratify=[labels[p] for p in valid_proteins], random_state=config.SEED)
    
    # 4. Features
    esm_extractor = BalancedESMExtractor(config.DEVICE)
    esm_features = esm_extractor.extract(valid_proteins, sequences, config.ESM_BATCH_SIZE)
    
    seq_extractor = BalancedFeatureExtractor()
    basic_features = {p: seq_extractor.extract(sequences[p]) for p in tqdm(valid_proteins, desc="Basic Feat")}
    
    all_basic = np.array([basic_features[p] for p in valid_proteins])
    scaler = RobustScaler()
    all_basic_norm = scaler.fit_transform(all_basic)
    
    seq_features = {}
    for i, p in enumerate(valid_proteins):
        seq_features[p] = torch.cat([esm_features[p], torch.tensor(all_basic_norm[i], dtype=torch.float32)])
    seq_dim = seq_features[valid_proteins[0]].shape[0]
    
    # 5. Graphs
    print("Building graphs...")
    graph_feat_extractor = BalancedGraphFeatures(adjacency)
    graphs = {p: build_balanced_subgraph(p, adjacency, seq_features, graph_feat_extractor, config.MAX_NEIGHBORS)
              for p in tqdm(valid_proteins, desc="Graphs")}
    node_dim = graphs[valid_proteins[0]].x.shape[1]
    
    # 6. Training
    kfold = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)
    train_val_labels = [labels[p] for p in train_val_proteins]
    fold_results, fold_weights = [], []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_val_proteins, train_val_labels)):
        print(f"\n>>> Fold {fold+1}")
        train_proteins = [train_val_proteins[i] for i in train_idx]
        val_proteins = [train_val_proteins[i] for i in val_idx]
        
        train_ds = RBPDataset(train_proteins, graphs, seq_features, labels)
        val_ds = RBPDataset(val_proteins, graphs, seq_features, labels)
        
        
        sample_weights = [neg_count/pos_count * 0.9 if labels[p] == 1 else 1.0 for p in train_proteins]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        
        train_loader = DataLoader(train_ds, batch_size=config.TRAIN_BATCH_SIZE, sampler=sampler, collate_fn=collate_fn, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=config.VAL_BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)
        
        model = BalancedGATModel(
            node_dim=node_dim, seq_dim=seq_dim,
            hidden_dim=config.HIDDEN_DIM, gat_hidden=config.GAT_HIDDEN,
            gat_heads=config.GAT_HEADS, num_layers=config.GAT_LAYERS,
            dropout=config.DROPOUT, use_edge_attr=config.USE_EDGE_ATTR
        ).to(config.DEVICE)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        scheduler = WarmupCosineScheduler(optimizer, config.WARMUP_EPOCHS, config.EPOCHS, config.MIN_LR)
        criterion = FocalLossWithSmoothing(alpha=config.FOCAL_ALPHA, gamma=config.FOCAL_GAMMA, smoothing=config.LABEL_SMOOTHING)
        trainer = BalancedTrainer(model, config.DEVICE, criterion, optimizer, use_amp=config.USE_AMP, use_mixup=config.USE_MIXUP, mixup_alpha=config.MIXUP_ALPHA)
        
        swa_model = AveragedModel(model) if config.USE_SWA else None
        swa_scheduler = SWALR(optimizer, swa_lr=config.SWA_LR) if config.USE_SWA else None
        
        best_val_auc = 0
        best_state = None
        patience = 0
        
        for epoch in range(1, config.EPOCHS + 1):
            lr = scheduler.step(epoch - 1)
            train_loss, train_auc = trainer.train_epoch(train_loader)
            val_metrics = trainer.evaluate(val_loader)
            
            if config.USE_SWA and epoch >= config.SWA_START:
                swa_model.update_parameters(model)
                swa_scheduler.step()
            
            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else: patience += 1
            
            if epoch % 5 == 0 or patience == 0:
                print(f"  Ep {epoch:2d} | Train: {train_auc:.4f} | Val: {val_metrics['auc']:.4f}")
            
            if patience >= config.PATIENCE: break
        
        if config.USE_SWA:
            torch.optim.swa_utils.update_bn(train_loader, swa_model, device=config.DEVICE)
            swa_metrics = trainer.evaluate(val_loader)
            if swa_metrics['auc'] >= best_val_auc - 0.003: 
                best_state = {k: v.cpu().clone() for k, v in swa_model.module.state_dict().items()}
                print(f"  SWA Selected (AUC: {swa_metrics['auc']:.4f})")
        
        model.load_state_dict(best_state)
        final_metrics = trainer.evaluate(val_loader, use_tta=config.USE_TTA)
        fold_results.append(final_metrics)
        fold_weights.append(final_metrics['auc'])
        print(f"  Result: {final_metrics['auc']:.4f}")
        torch.save(best_state, f'models/balanced_fold_{fold+1}.pt')
        clear_memory()

    # 7. Summary
    cv_aucs = [r['auc'] for r in fold_results]
    print(f"\nCV AUC: {np.mean(cv_aucs):.4f}")
    
    # 8. Test
    print("\nTesting...")
    test_ds = RBPDataset(test_proteins, graphs, seq_features, labels)
    test_loader = DataLoader(test_ds, batch_size=config.VAL_BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    weights = np.array(fold_weights)
    weights = np.exp(weights * 6) / np.sum(np.exp(weights * 6)) # 稍微加大权重的锐度
    
    all_preds = []
    for fold in range(config.N_FOLDS):
        model = BalancedGATModel(
            node_dim=node_dim, seq_dim=seq_dim,
            hidden_dim=config.HIDDEN_DIM, gat_hidden=config.GAT_HIDDEN,
            gat_heads=config.GAT_HEADS, num_layers=config.GAT_LAYERS,
            dropout=config.DROPOUT, use_edge_attr=config.USE_EDGE_ATTR
        ).to(config.DEVICE)
        model.load_state_dict(torch.load(f'models/balanced_fold_{fold+1}.pt'))
        model.eval()
        
        fold_p = []
        with torch.no_grad():
            for batch in test_loader:
                graph = batch['graph'].to(config.DEVICE)
                seq = batch['seq_features'].to(config.DEVICE)
                
                if config.USE_TTA:
                    ps = [torch.sigmoid(model(graph, seq, training=False)[0].float())]
                    for _ in range(8):
                        out, _ = model(graph, seq + torch.randn_like(seq)*0.02, training=False)
                        ps.append(torch.sigmoid(out.float()))
                    p = torch.stack(ps).mean(dim=0)
                else:
                    p = torch.sigmoid(model(graph, seq, training=False)[0].float())
                fold_p.extend(p.cpu().numpy())
        all_preds.append(fold_p)
    
    final_preds = np.average(all_preds, axis=0, weights=weights)
    test_labels = np.array([labels[p] for p in test_proteins])
    test_auc = roc_auc_score(test_labels, final_preds)
    
    print("\n" + "="*40)
    print(f"FINAL TEST AUC: {test_auc:.4f}")
    print("="*40)
    
    with open('results/balanced_final_results.json', 'w') as f:
        json.dump({'test_auc': test_auc}, f)

if __name__ == "__main__":
    main()
