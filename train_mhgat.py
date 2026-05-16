import os
import json
import math
import random
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from torch_geometric.nn import GATv2Conv, HeteroConv
from torch_geometric.data import HeteroData
import esm
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    rna_embed_dim: int = 128
    protein_embed_dim: int = 1280
    hidden_dim: int = 256
    output_dim: int = 128
    graph_scales: Tuple[int, ...] = (3, 7, 15)
    num_gat_layers: int = 3
    num_heads: int = 8
    gat_dropout: float = 0.1
    num_transformer_layers: int = 4
    transformer_heads: int = 8
    transformer_dropout: float = 0.1
    region_size: int = 10
    contrastive_temp: float = 0.07
    contrastive_weight: float = 0.5
    max_rna_len: int = 512
    max_protein_len: int = 1024
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 100
    warmup_epochs: int = 5
    esm_model_name: str = "esm2_t33_650M_UR50D"
    freeze_esm: bool = True
    device: str = "cuda"

@dataclass
class TrainingConfig:
    qbiolip_data: str = "data/qbiolip/processed/qbiolip_unified.json"
    arabidopsis_data: str = "data/arabidopsis/processed/arabidopsis_unified.json"
    output_dir: str = "outputs/mhgat"
    pretrain_epochs: int = 20
    finetune_epochs: int = 80
    transfer_epochs: int = 50
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    save_every: int = 5
    eval_every: int = 1
    seed: int = 42
    num_workers: int = 4
    use_wandb: bool = False

class RNATokenizer:
    VOCAB = {
        '<pad>': 0, '<unk>': 1, '<cls>': 2, '<sep>': 3, '<mask>': 4,
        'A': 5, 'U': 6, 'G': 7, 'C': 8, 'N': 9, 'I': 10, 'T': 11
    }
    
    def __init__(self, max_len: int = 512):
        self.max_len = max_len
        self.vocab_size = len(self.VOCAB)
    
    def encode(self, sequence: str) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence = sequence.upper().replace('T', 'U')
        if len(sequence) > self.max_len - 2:
            sequence = sequence[:self.max_len - 2]
        tokens = [self.VOCAB['<cls>']]
        for char in sequence:
            tokens.append(self.VOCAB.get(char, self.VOCAB['<unk>']))
        tokens.append(self.VOCAB['<sep>'])
        seq_len = len(tokens)
        attention_mask = [1] * seq_len
        while len(tokens) < self.max_len:
            tokens.append(self.VOCAB['<pad>'])
            attention_mask.append(0)
        return (torch.tensor(tokens, dtype=torch.long), torch.tensor(attention_mask, dtype=torch.long))

class BindingSiteDataset(Dataset):
    def __init__(self, data_file: str, rna_tokenizer: RNATokenizer, esm_alphabet, max_rna_len: int = 512, max_protein_len: int = 1024, augment: bool = False):
        self.rna_tokenizer = rna_tokenizer
        self.esm_alphabet = esm_alphabet
        self.max_rna_len = max_rna_len
        self.max_protein_len = max_protein_len
        self.augment = augment
        with open(data_file, 'r') as f:
            data = json.load(f)
        self.samples = data['samples']
        self.metadata = data.get('metadata', {})
        logger.info(f"Loading dataset: {data_file}")
        logger.info(f"  Samples: {len(self.samples)}")
        self.esm_batch_converter = self.esm_alphabet.get_batch_converter()
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        rna_seq, protein_seq, binding_labels = sample['rna_sequence'], sample['protein_sequence'], sample['binding_labels']
        if self.augment:
            rna_seq, binding_labels = self._augment(rna_seq, binding_labels)
        rna_tokens, rna_mask = self.rna_tokenizer.encode(rna_seq)
        protein_seq_truncated = protein_seq[:self.max_protein_len]
        _, _, protein_tokens = self.esm_batch_converter([("protein", protein_seq_truncated)])
        protein_tokens = protein_tokens.squeeze(0)
        protein_mask = (protein_tokens != self.esm_alphabet.padding_idx).long()
        actual_rna_len = min(len(rna_seq), self.max_rna_len - 2)
        labels = [0] + binding_labels[:actual_rna_len] + [0]
        while len(labels) < self.max_rna_len:
            labels.append(-100)
        return {
            'rna_tokens': rna_tokens, 'rna_mask': rna_mask, 'protein_tokens': protein_tokens,
            'protein_mask': protein_mask, 'labels': torch.tensor(labels, dtype=torch.long),
            'rna_len': actual_rna_len, 'protein_len': len(protein_seq_truncated),
            'sample_id': sample.get('sample_id', f'sample_{idx}')
        }
    
    def _augment(self, rna_seq: str, labels: List[int]) -> Tuple[str, List[int]]:
        if random.random() < 0.1:
            seq_list = list(rna_seq)
            for i in range(len(seq_list)):
                if random.random() < 0.02:
                    seq_list[i] = random.choice(['A', 'U', 'G', 'C'])
            rna_seq = ''.join(seq_list)
        return rna_seq, labels

def collate_fn(batch: List[Dict]) -> Dict:
    max_protein_len = max(b['protein_tokens'].size(0) for b in batch)
    padded_protein, padded_protein_mask = [], []
    for b in batch:
        curr_len = b['protein_tokens'].size(0)
        if curr_len < max_protein_len:
            pad = torch.zeros(max_protein_len - curr_len, dtype=torch.long)
            padded_protein.append(torch.cat([b['protein_tokens'], pad]))
            padded_protein_mask.append(torch.cat([b['protein_mask'], pad]))
        else:
            padded_protein.append(b['protein_tokens'])
            padded_protein_mask.append(b['protein_mask'])
    return {
        'rna_tokens': torch.stack([b['rna_tokens'] for b in batch]),
        'rna_mask': torch.stack([b['rna_mask'] for b in batch]),
        'protein_tokens': torch.stack(padded_protein),
        'protein_mask': torch.stack(padded_protein_mask),
        'labels': torch.stack([b['labels'] for b in batch]),
        'rna_len': [b['rna_len'] for b in batch],
        'protein_len': [b['protein_len'] for b in batch],
        'sample_id': [b['sample_id'] for b in batch]
    }

class RNAEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, embed_dim)
        self.structure_embedding = nn.Embedding(max_len, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        nn.init.normal_(self.structure_embedding.weight, std=0.01)
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, L = tokens.shape
        positions = torch.arange(L, device=tokens.device).unsqueeze(0).expand(B, -1)
        embeddings = (self.token_embedding(tokens) + self.position_embedding(positions) + self.structure_embedding(positions) * 0.1)
        return self.dropout(self.layer_norm(embeddings))

class ESM2Encoder(nn.Module):
    def __init__(self, model_name: str = "esm2_t33_650M_UR50D", freeze: bool = True):
        super().__init__()
        logger.info(f"Loading ESM2 model: {model_name}")
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        if freeze:
            for param in self.model.parameters(): param.requires_grad = False
            logger.info("ESM2 parameters frozen")
        self.output_dim = self.model.embed_dim
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        with torch.no_grad() if not self.training else torch.enable_grad():
            results = self.model(tokens, repr_layers=[33], return_contacts=False)
            return results["representations"][33]

class MultiScaleGraphBuilder(nn.Module):
    def __init__(self, scales: Tuple[int, ...] = (3, 7, 15)):
        super().__init__()
        self.scales = scales
    
    def build_sequence_graph(self, seq_len: int, scale: int) -> Tuple[torch.Tensor, torch.Tensor]:
        edges, weights = [], []
        for i in range(seq_len):
            for j in range(max(0, i - scale), min(seq_len, i + scale + 1)):
                if i != j:
                    edges.append([i, j])
                    weights.append(1.0 / (1.0 + abs(i - j)))
        if not edges: return (torch.zeros(2, 0, dtype=torch.long), torch.zeros(0))
        return (torch.tensor(edges, dtype=torch.long).t().contiguous(), torch.tensor(weights, dtype=torch.float))
    
    def forward(self, seq_lens: List[int], device: torch.device) -> Dict[str, List]:
        graphs = {scale: [] for scale in self.scales}
        for seq_len in seq_lens:
            for scale in self.scales:
                edge_idx, edge_attr = self.build_sequence_graph(seq_len, scale)
                graphs[scale].append((edge_idx.to(device), edge_attr.to(device)))
        return graphs

class DynamicEdgeWeight(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.edge_mlp = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1), nn.Sigmoid())
    
    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor, base_weights: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        edge_features = torch.cat([node_features[src], node_features[dst]], dim=-1)
        return base_weights * self.edge_mlp(edge_features).squeeze(-1)

class MultiScaleGAT(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_heads: int = 8, scales: Tuple[int, ...] = (3, 7, 15), dropout: float = 0.1):
        super().__init__()
        self.scales = scales
        self.gat_layers = nn.ModuleDict({str(s): GATv2Conv(in_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout, edge_dim=1, add_self_loops=True) for s in scales})
        self.edge_weight_learners = nn.ModuleDict({str(s): DynamicEdgeWeight(in_dim) for s in scales})
        self.scale_fusion = nn.Sequential(nn.Linear(hidden_dim * len(scales), hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, out_dim))
        self.scale_attention = nn.Sequential(nn.Linear(hidden_dim * len(scales), len(scales)), nn.Softmax(dim=-1))
    
    def forward(self, node_features: torch.Tensor, multi_scale_graphs: Dict[int, Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        scale_outputs = []
        for scale in self.scales:
            sk = str(scale)
            idx, attr = multi_scale_graphs[scale]
            dyn_attr = self.edge_weight_learners[sk](node_features, idx, attr)
            scale_outputs.append(self.gat_layers[sk](node_features, idx, edge_attr=dyn_attr.unsqueeze(-1)))
        concat_features = torch.cat(scale_outputs, dim=-1)
        stacked = torch.stack(scale_outputs, dim=-1)
        weights = self.scale_attention(concat_features)
        weighted_sum = (stacked * weights.unsqueeze(1)).sum(dim=-1)
        return self.scale_fusion(concat_features) + weighted_sum

class HeterogeneousGraphBuilder(nn.Module):
    def __init__(self, rna_dim: int, protein_dim: int, hidden_dim: int, num_cross_edges: int = 32):
        super().__init__()
        self.num_cross_edges = num_cross_edges
        self.rna_proj = nn.Linear(rna_dim, hidden_dim)
        self.protein_proj = nn.Linear(protein_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, dropout=0.1, batch_first=True)
    
    def forward(self, rna_features: torch.Tensor, protein_features: torch.Tensor, rna_mask: torch.Tensor, protein_mask: torch.Tensor) -> List[HeteroData]:
        B = rna_features.size(0)
        rh, ph = self.rna_proj(rna_features), self.protein_proj(protein_features)
        _, weights = self.cross_attn(rh, ph, ph, key_padding_mask=(protein_mask == 0))
        h_graphs = []
        for b in range(B):
            data = HeteroData()
            rl, pl = rna_mask[b].sum().item(), protein_mask[b].sum().item()
            data['rna'].x, data['protein'].x = rh[b, :rl], ph[b, :pl]
            re = [[i, i+1] for i in range(rl-1)] + [[i+1, i] for i in range(rl-1)]
            pe = [[i, i+1] for i in range(pl-1)] + [[i+1, i] for i in range(pl-1)]
            if re: data['rna', 'connects', 'rna'].edge_index = torch.tensor(re, device=rna_features.device).t().contiguous()
            if pe: data['protein', 'connects', 'protein'].edge_index = torch.tensor(pe, device=rna_features.device).t().contiguous()
            attn = weights[b, :rl, :pl]
            flat = attn.flatten()
            ne = min(self.num_cross_edges, flat.numel())
            _, top = flat.topk(ne)
            ri, pi = top // pl, top % pl
            data['rna', 'binds', 'protein'].edge_index = torch.stack([ri, pi])
            data['rna', 'binds', 'protein'].edge_attr = flat[top].unsqueeze(-1)
            data['protein', 'bound_by', 'rna'].edge_index = torch.stack([pi, ri])
            data['protein', 'bound_by', 'rna'].edge_attr = flat[top].unsqueeze(-1)
            h_graphs.append(data)
        return h_graphs

class HeterogeneousGAT(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(HeteroConv({
                ('rna', 'connects', 'rna'): GATv2Conv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout, add_self_loops=True),
                ('protein', 'connects', 'protein'): GATv2Conv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout, add_self_loops=True),
                ('rna', 'binds', 'protein'): GATv2Conv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout, edge_dim=1, add_self_loops=False),
                ('protein', 'bound_by', 'rna'): GATv2Conv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout, edge_dim=1, add_self_loops=False),
            }, aggr='sum'))
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, h_data: HeteroData) -> Dict[str, torch.Tensor]:
        x_dict = {'rna': h_data['rna'].x, 'protein': h_data['protein'].x}
        e_idx_dict = {k: h_data[k].edge_index for k in h_data.edge_types if hasattr(h_data[k], 'edge_index')}
        e_attr_dict = {k: h_data[k].edge_attr for k in h_data.edge_types if hasattr(h_data[k], 'edge_attr')}
        for layer in self.layers:
            new_x = layer(x_dict, e_idx_dict, e_attr_dict)
            for k in new_x:
                if k in x_dict: new_x[k] = self.layer_norm(x_dict[k] + self.dropout(new_x[k]))
            x_dict = new_x
        return x_dict

class StructureGuidedTransformer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 8, num_layers: int = 4, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.hidden_dim, self.num_heads, self.max_len = hidden_dim, num_heads, max_len
        self.rel_pos_bias = nn.Parameter(torch.zeros(num_heads, 2 * max_len - 1))
        nn.init.normal_(self.rel_pos_bias, std=0.02)
        ly = nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * 4, dropout, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(ly, num_layers=num_layers)
        self.bias_mlp = nn.Sequential(nn.Linear(hidden_dim, num_heads), nn.Tanh())
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        m = (mask == 0) if mask is not None else None
        return self.encoder(x, src_key_padding_mask=m)

class HierarchicalPredictor(nn.Module):
    def __init__(self, hidden_dim: int, region_size: int = 10, dropout: float = 0.1):
        super().__init__()
        self.region_size = region_size
        self.region_conv = nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=region_size, stride=region_size // 2, padding=region_size // 2)
        self.region_predictor = nn.Sequential(nn.LayerNorm(hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1))
        self.base_predictor = nn.Sequential(nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 2))
        self.upsample = nn.Upsample(scale_factor=region_size // 2, mode='linear', align_corners=True)
    
    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = h.shape
        rf = self.region_predictor(self.region_conv(h.permute(0, 2, 1)).permute(0, 2, 1))
        ru = self.upsample(self.region_conv(h.permute(0, 2, 1)))
        if ru.size(2) > L: ru = ru[:, :, :L]
        elif ru.size(2) < L: ru = torch.cat([ru, torch.zeros(B, ru.size(1), L - ru.size(2), device=h.device)], dim=2)
        bl = self.base_predictor(torch.cat([h, ru.permute(0, 2, 1)], dim=-1))
        return bl, rf

class ContrastiveLearner(nn.Module):
    def __init__(self, hidden_dim: int, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.rna_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim // 2))
        self.prot_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim // 2))
    
    def forward(self, rf: torch.Tensor, pf: torch.Tensor, rm: torch.Tensor, pm: torch.Tensor) -> torch.Tensor:
        B = rf.size(0)
        rmf, pmf = rm.unsqueeze(-1).float(), pm.unsqueeze(-1).float()
        rp = F.normalize(self.rna_proj((rf * rmf).sum(1) / rmf.sum(1).clamp(min=1)), dim=-1)
        pp = F.normalize(self.prot_proj((pf * pmf).sum(1) / pmf.sum(1).clamp(min=1)), dim=-1)
        sim = torch.matmul(rp, pp.t()) / self.temperature
        lbl = torch.arange(B, device=sim.device)
        return (F.cross_entropy(sim, lbl) + F.cross_entropy(sim.t(), lbl)) / 2

class MHGAT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.rna_embedding = RNAEmbedding(RNATokenizer().vocab_size, config.rna_embed_dim, config.max_rna_len, config.gat_dropout)
        self.protein_encoder = ESM2Encoder(config.esm_model_name, config.freeze_esm)
        self.rna_align = nn.Linear(config.rna_embed_dim, config.hidden_dim)
        self.protein_align = nn.Linear(config.protein_embed_dim, config.hidden_dim)
        self.graph_builder = MultiScaleGraphBuilder(scales=config.graph_scales)
        self.multiscale_gat = MultiScaleGAT(config.hidden_dim, config.hidden_dim, config.hidden_dim, config.num_heads, config.graph_scales, config.gat_dropout)
        self.hetero_graph_builder = HeterogeneousGraphBuilder(config.hidden_dim, config.hidden_dim, config.hidden_dim)
        self.hetero_gat = HeterogeneousGAT(config.hidden_dim, 4, 2, config.gat_dropout)
        self.structure_transformer = StructureGuidedTransformer(config.hidden_dim, config.transformer_heads, config.num_transformer_layers, config.transformer_dropout, config.max_rna_len)
        self.hierarchical_predictor = HierarchicalPredictor(config.hidden_dim, config.region_size, config.gat_dropout)
        self.contrastive_learner = ContrastiveLearner(config.hidden_dim, config.contrastive_temp)
        self.fusion = nn.Sequential(nn.Linear(config.hidden_dim * 3, config.hidden_dim), nn.LayerNorm(config.hidden_dim), nn.ReLU(), nn.Dropout(config.gat_dropout))
    
    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        device, B = batch['rna_tokens'].device, batch['rna_tokens'].size(0)
        re = self.rna_embedding(batch['rna_tokens'])
        pe = self.protein_encoder(batch['protein_tokens'])
        rh, ph = self.rna_align(re), self.protein_align(pe)
        rlens = [batch['rna_mask'][b].sum().item() for b in range(B)]
        ms_graphs = self.graph_builder(rlens, device)
        all_rf = [rh[b, :rlens[b]] for b in range(B)]
        if all_rf:
            flat_rf = torch.cat(all_rf, dim=0)
            merged = {}
            for s in self.config.graph_scales:
                ae, aw, off = [], [], 0
                for b in range(B):
                    idx, att = ms_graphs[s][b]
                    ae.append(idx + off)
                    aw.append(att)
                    off += rlens[b]
                if ae and any(e.numel() > 0 for e in ae):
                    ve, vw = [e for e in ae if e.numel() > 0], [w for w in aw if w.numel() > 0]
                    merged[s] = (torch.cat(ve, dim=1), torch.cat(vw)) if ve else (torch.zeros(2, 0, dtype=torch.long, device=device), torch.zeros(0, device=device))
                else: merged[s] = (torch.zeros(2, 0, dtype=torch.long, device=device), torch.zeros(0, device=device))
            gat_out = self.multiscale_gat(flat_rf, merged)
            gf, off = torch.zeros_like(rh), 0
            for b in range(B):
                gf[b, :rlens[b]], off = gat_out[off:off + rlens[b]], off + rlens[b]
        else: gf = rh
        h_graphs = self.hetero_graph_builder(rh, ph, batch['rna_mask'], batch['protein_mask'])
        hrf = torch.zeros_like(rh)
        for b, h_data in enumerate(h_graphs):
            if hasattr(h_data['rna'], 'x') and h_data['rna'].x.size(0) > 0:
                nodes = self.hetero_gat(h_data)
                hrf[b, :nodes['rna'].size(0)] = nodes['rna']
        fused = self.fusion(torch.cat([rh, gf, hrf], dim=-1))
        tr_out = self.structure_transformer(fused, batch['rna_mask'])
        bl, rl = self.hierarchical_predictor(tr_out)
        cl = self.contrastive_learner(fused, ph, batch['rna_mask'], batch['protein_mask'])
        return {'base_logits': bl, 'region_logits': rl, 'contrastive_loss': cl, 'rna_features': fused}

class MHGATTrainer:
    def __init__(self, model, model_config, training_config):
        self.model, self.model_config, self.training_config = model, model_config, training_config
        self.device = torch.device(model_config.device if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=model_config.learning_rate, weight_decay=model_config.weight_decay)
        self.scheduler, self.scaler = None, GradScaler() if torch.cuda.is_available() else None
        self.output_dir = Path(training_config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_metrics = {'val_auc': 0.0}
    
    def compute_loss(self, outputs, batch, include_contrastive=True):
        bl, lbl = outputs['base_logits'], batch['labels'].to(self.device)
        v_mask = lbl != -100
        ce = F.cross_entropy(bl[v_mask], lbl[v_mask], weight=torch.tensor([1.0, 3.0], device=self.device)) if v_mask.sum() > 0 else torch.tensor(0.0, device=self.device)
        rl = outputs['region_logits']
        rs, B, L = self.model_config.region_size, lbl.shape[0], lbl.shape[1]
        nr = rl.size(1)
        rlbl = torch.zeros(B, nr, device=self.device)
        for b in range(B):
            for r in range(nr):
                start, end = r * (rs // 2), min(r * (rs // 2) + rs, L)
                if end <= L:
                    slc = lbl[b, start:end]
                    v_reg = slc[slc != -100]
                    if v_reg.numel() > 0 and v_reg.sum() > 0: rlbl[b, r] = 1.0
        reg_loss = F.binary_cross_entropy_with_logits(rl.squeeze(-1), rlbl)
        tot = ce + 0.3 * reg_loss
        if include_contrastive: tot += self.model_config.contrastive_weight * outputs['contrastive_loss']
        return {'total_loss': tot, 'ce_loss': ce, 'region_loss': reg_loss, 'contrastive_loss': outputs['contrastive_loss']}
    
    def train_epoch(self, loader, epoch, include_contrastive=True):
        self.model.train()
        t_losses, nb = defaultdict(float), 0
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            self.optimizer.zero_grad()
            if self.scaler:
                with autocast():
                    outputs = self.model(batch)
                    losses = self.compute_loss(outputs, batch, include_contrastive)
                self.scaler.scale(losses['total_loss']).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(batch)
                losses = self.compute_loss(outputs, batch, include_contrastive)
                losses['total_loss'].backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            if self.scheduler: self.scheduler.step()
            for k, v in losses.items(): t_losses[k] += v.item()
            nb += 1
            pbar.set_postfix({'loss': f"{losses['total_loss'].item():.4f}"})
        return {k: v / nb for k, v in t_losses.items()}

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        ap, al, apr, tl, nb = [], [], [], 0.0, 0
        for batch in tqdm(loader, desc="Evaluating"):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = self.model(batch)
            tl += self.compute_loss(outputs, batch, False)['total_loss'].item()
            nb += 1
            bl, lbl = outputs['base_logits'], batch['labels']
            prb, v_mask = F.softmax(bl, dim=-1)[:, :, 1], lbl != -100
            apr.extend(prb[v_mask].cpu().numpy())
            ap.extend(bl.argmax(dim=-1)[v_mask].cpu().numpy())
            al.extend(lbl[v_mask].cpu().numpy())
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
        ap, al, apr = np.array(ap), np.array(al), np.array(apr)
        met = {'loss': tl/nb, 'accuracy': accuracy_score(al, ap), 'precision': precision_score(al, ap, zero_division=0), 'recall': recall_score(al, ap, zero_division=0), 'f1': f1_score(al, ap, zero_division=0)}
        if len(np.unique(al)) > 1:
            met['auc'], met['auprc'] = roc_auc_score(al, apr), average_precision_score(al, apr)
        else: met['auc'] = met['auprc'] = 0.0
        return met

    def train(self, train_loader, val_loader, num_epochs, stage="pretrain"):
        logger.info(f"Starting training: {stage}")
        ts = len(train_loader) * num_epochs
        ws = len(train_loader) * self.model_config.warmup_epochs
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.model_config.learning_rate, total_steps=ts, pct_start=ws/ts, anneal_strategy='cos')
        ic = (stage == "pretrain")
        for ep in range(1, num_epochs + 1):
            tl = self.train_epoch(train_loader, ep, ic)
            if ep % self.training_config.eval_every == 0:
                vm = self.evaluate(val_loader)
                if vm['auc'] > self.best_metrics['val_auc']:
                    self.best_metrics['val_auc'] = vm['auc']
                    self.save_checkpoint(f"best_{stage}.pt", vm)
            if ep % self.training_config.save_every == 0: self.save_checkpoint(f"{stage}_epoch_{ep}.pt")

    def save_checkpoint(self, filename, metrics=None):
        torch.save({'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'model_config': self.model_config, 'best_metrics': self.best_metrics, 'metrics': metrics}, self.output_dir / filename)

class MHGATPredictor:
    def __init__(self, model, config, device="cuda"):
        self.model, self.config, self.device = model, config, torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.tok = RNATokenizer(config.max_rna_len)
        _, self.esm_alpha = esm.pretrained.load_model_and_alphabet(config.esm_model_name)
        self.bc = self.esm_alpha.get_batch_converter()
    
    @torch.no_grad()
    def predict(self, rna, prot):
        rt, rm = self.tok.encode(rna)
        ps = prot[:self.config.max_protein_len]
        _, _, pt = self.bc([("protein", ps)])
        pt, pm = pt.squeeze(0), (pt.squeeze(0) != self.esm_alpha.padding_idx).long()
        b = {k: v.unsqueeze(0).to(self.device) for k, v in {'rna_tokens': rt, 'rna_mask': rm, 'protein_tokens': pt, 'protein_mask': pm}.items()}
        out = self.model(b)
        prb = F.softmax(out['base_logits'][0], dim=-1)[:, 1].cpu().numpy()
        alen = min(len(rna), self.config.max_rna_len - 2)
        p_list = prb[1:alen+1].tolist()
        return {'probs': p_list, 'labels': [1 if p > 0.5 else 0 for p in p_list], 'rna_len': alen}

class TransferLearner:
    def __init__(self, model, config, t_config):
        self.model, self.config, self.t_config, self.device = model, config, t_config, torch.device(config.device if torch.cuda.is_available() else "cpu")
    
    def transfer(self, source_ckpt, target_train, target_val, num_epochs=50, freeze=['protein_encoder', 'rna_embedding']):
        checkpoint = torch.load(source_ckpt, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        for n, p in self.model.named_parameters():
            if any(f in n for f in freeze): p.requires_grad = False
        trainer = MHGATTrainer(self.model, self.config, self.t_config)
        for pg in trainer.optimizer.param_groups: pg['lr'] *= 0.1
        trainer.train(target_train, target_val, num_epochs, "transfer")

def prepare_data(t_config, m_config):
    _, alpha = esm.pretrained.load_model_and_alphabet(m_config.esm_model_name)
    tok = RNATokenizer(m_config.max_rna_len)
    res = {}
    for key, path in [('qbiolip', t_config.qbiolip_data), ('arabidopsis', t_config.arabidopsis_data)]:
        if os.path.exists(path):
            ds = BindingSiteDataset(path, tok, alpha, m_config.max_rna_len, m_config.max_protein_len, True)
            tr_sz = int(len(ds) * t_config.train_ratio)
            vl_sz = int(len(ds) * t_config.val_ratio)
            tr, vl, ts = random_split(ds, [tr_sz, vl_sz, len(ds) - tr_sz - vl_sz], generator=torch.Generator().manual_seed(t_config.seed))
            res[key] = {'train': tr, 'val': vl, 'test': ts}
    return res

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--qbiolip_data', default='data/qbiolip/processed/qbiolip_unified.json')
    p.add_argument('--arabidopsis_data', default='data/arabidopsis/processed/arabidopsis_unified.json')
    p.add_argument('--output_dir', default='outputs/mhgat')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--learning_rate', type=float, default=1e-4)
    p.add_argument('--pretrain_epochs', type=int, default=20)
    p.add_argument('--finetune_epochs', type=int, default=80)
    p.add_argument('--transfer_epochs', type=int, default=50)
    p.add_argument('--hidden_dim', type=int, default=256)
    p.add_argument('--num_gat_layers', type=int, default=3)
    p.add_argument('--num_heads', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda')
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--mode', choices=['train', 'transfer', 'predict'], default='train')
    p.add_argument('--checkpoint', default=None)
    args = p.parse_args()
    
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    m_cfg = ModelConfig(hidden_dim=args.hidden_dim, num_gat_layers=args.num_gat_layers, num_heads=args.num_heads, batch_size=args.batch_size, learning_rate=args.learning_rate, device=args.device)
    t_cfg = TrainingConfig(qbiolip_data=args.qbiolip_data, arabidopsis_data=args.arabidopsis_data, output_dir=args.output_dir, pretrain_epochs=args.pretrain_epochs, finetune_epochs=args.finetune_epochs, transfer_epochs=args.transfer_epochs, seed=args.seed, num_workers=args.num_workers)
    
    ds = prepare_data(t_cfg, m_cfg)
    model = MHGAT(m_cfg)
    
    if args.mode == 'train' and 'qbiolip' in ds:
        ldr = {k: DataLoader(ds['qbiolip'][k], batch_size=m_cfg.batch_size, shuffle=(k=='train'), num_workers=t_cfg.num_workers, collate_fn=collate_fn) for k in ['train', 'val', 'test']}
        tr = MHGATTrainer(model, m_cfg, t_cfg)
        tr.train(ldr['train'], ldr['val'], t_cfg.pretrain_epochs, "pretrain")
        tr.train(ldr['train'], ldr['val'], t_cfg.finetune_epochs, "finetune")
        logger.info(f"Test Metrics: {tr.evaluate(ldr['test'])}")
    elif args.mode == 'transfer' and args.checkpoint and 'arabidopsis' in ds:
        ldr = {k: DataLoader(ds['arabidopsis'][k], batch_size=m_cfg.batch_size, shuffle=(k=='train'), num_workers=t_cfg.num_workers, collate_fn=collate_fn) for k in ['train', 'val']}
        TransferLearner(model, m_cfg, t_cfg).transfer(args.checkpoint, ldr['train'], ldr['val'], t_cfg.transfer_epochs)
    elif args.mode == 'predict' and args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=m_cfg.device)['model_state_dict'])
        res = MHGATPredictor(model, m_cfg).predict("AUGCUAGCUAGCUAGCUAGCUAGCUAGCUAG", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLA")
        logger.info(f"Result: {res}")

if __name__ == "__main__":
    main()
