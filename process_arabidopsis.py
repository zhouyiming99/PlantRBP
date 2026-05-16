import os
import re
import gzip
import json
import numpy as np
import pandas as pd
import pysam
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BindingPeak:
    chrom: str
    start: int
    end: int
    strand: str
    rbp_name: str
    method: str
    tissue: str
    accession: str
    score: float
    peak_id: str = ""

@dataclass
class ArabidopsisBindingSample:
    rna_sequence: str
    protein_sequence: str
    binding_labels: List[int]
    sample_id: str
    source: str = "arabidopsis_clip"
    rbp_name: str = ""
    gene_id: str = ""
    chrom: str = ""
    start: int = 0
    end: int = 0
    strand: str = "+"
    method: str = ""
    tissue: str = ""
    score: float = 0.0

class GTFParser:
    def __init__(self, gtf_file: str):
        self.gtf_file = gtf_file
        self.symbol_to_gene_id = {}      
        self.gene_id_to_symbol = {}      
        self.gene_id_to_name = {}        
        self.transcript_to_gene = {}     
        self._parse()
    
    def _parse(self):
        logger.info(f"Parsing GTF file: {self.gtf_file}")
        opener = gzip.open if self.gtf_file.endswith('.gz') else open
        mode = 'rt' if self.gtf_file.endswith('.gz') else 'r'
        
        gene_count = 0
        with opener(self.gtf_file, mode) as f:
            for line in f:
                if line.startswith('#'): continue
                parts = line.strip().split('\t')
                if len(parts) < 9: continue
                
                feature_type = parts[2]
                attributes = parts[8]
                attr_dict = self._parse_attributes(attributes)
                gene_id = attr_dict.get('gene_id', '').replace('"', '')
                if not gene_id: continue
                
                gene_symbol = attr_dict.get('gene_symbol', '').replace('"', '')
                gene_name = attr_dict.get('gene_name', '').replace('"', '')
                transcript_id = attr_dict.get('transcript_id', '').replace('"', '')
                description = attr_dict.get('description', '').replace('"', '')
                
                if not gene_symbol and description:
                    match = re.search(r'gene[_\s]?symbol[:\s]+(\w+)', description, re.I)
                    if match: gene_symbol = match.group(1)
                
                if not gene_symbol: gene_symbol = gene_name
                
                if gene_id:
                    gene_id_base = gene_id.split('.')[0]
                    if gene_symbol:
                        self.symbol_to_gene_id[gene_symbol.upper()] = gene_id_base
                        self.gene_id_to_symbol[gene_id_base] = gene_symbol
                    if gene_name:
                        self.symbol_to_gene_id[gene_name.upper()] = gene_id_base
                        self.gene_id_to_name[gene_id_base] = gene_name
                    if transcript_id:
                        self.transcript_to_gene[transcript_id.split('.')[0]] = gene_id_base
                    if feature_type == 'gene': gene_count += 1
        
        logger.info(f"Parsed {gene_count} genes")
        self._add_known_rbp_mappings()
    
    def _parse_attributes(self, attr_string: str) -> Dict[str, str]:
        attrs = {}
        for attr in attr_string.split(';'):
            attr = attr.strip()
            if not attr: continue
            match = re.match(r'(\w+)\s+"?([^"]*)"?', attr)
            if match:
                attrs[match.group(1)] = match.group(2)
            elif '=' in attr:
                key, value = attr.split('=', 1)
                attrs[key.strip()] = value.strip().strip('"')
        return attrs
    
    def _add_known_rbp_mappings(self):
        known_rbps = {
            'GRP7': 'AT2G21660', 'GRP8': 'AT4G39260', 'GR-RBP2': 'AT4G13850', 'GR-RBP3': 'AT5G61030',
            'FCA': 'AT4G16280', 'FPA': 'AT2G43410', 'SR34': 'AT1G02840', 'AGO1': 'AT1G48410',
            'SE': 'AT2G27100', 'HYL1': 'AT1G09700', 'PTB1': 'AT3G01150', 'PTB2': 'AT5G53180'
        }
        for symbol, gene_id in known_rbps.items():
            self.symbol_to_gene_id[symbol.upper()] = gene_id
            if gene_id not in self.gene_id_to_symbol: self.gene_id_to_symbol[gene_id] = symbol
    
    def get_gene_id(self, symbol: str) -> Optional[str]:
        s_upper = symbol.upper().strip()
        if s_upper in self.symbol_to_gene_id: return self.symbol_to_gene_id[s_upper]
        return None
    
    def get_symbol(self, gene_id: str) -> Optional[str]:
        return self.gene_id_to_symbol.get(gene_id.split('.')[0])

class ProteinSequenceManager:
    def __init__(self, fasta_file: str, gtf_parser: GTFParser = None):
        self.fasta_file = fasta_file
        self.gtf_parser = gtf_parser
        self.gene_to_sequence = {}       
        self.symbol_to_sequence = {}     
        self._parse()
    
    def _parse(self):
        logger.info(f"Parsing protein FASTA: {self.fasta_file}")
        current_id, current_gene, current_symbol = None, None, None
        current_seq = []
        with open(self.fasta_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_id and current_seq:
                        self._store_sequence(current_id, current_gene, current_symbol, ''.join(current_seq))
                    current_id, current_gene, current_symbol = self._parse_header(line)
                    current_seq = []
                elif line: current_seq.append(line)
        if current_id and current_seq:
            self._store_sequence(current_id, current_gene, current_symbol, ''.join(current_seq))
        logger.info(f"Loaded {len(self.gene_to_sequence)} protein sequences")
    
    def _parse_header(self, header: str) -> Tuple[str, str, str]:
        header = header[1:]
        tid = ""
        gid = ""
        sym = ""
        match = re.search(r'(?:lcl\|)?([A-Z]+\d+[A-Z]\d+(?:\.\d+)?)', header)
        if match: tid = match.group(1)
        match = re.search(r'gene[:\s]+([A-Z]+\d+[A-Z]\d+)', header)
        if match: gid = match.group(1)
        if not gid and tid: gid = tid.split('.')[0]
        match = re.search(r'gene_symbol[:\s]+(\w+)', header)
        if match: sym = match.group(1)
        if self.gtf_parser and gid:
            s_from_gtf = self.gtf_parser.get_symbol(gid)
            if s_from_gtf: sym = s_from_gtf
        return tid, gid, sym
    
    def _store_sequence(self, tid, gid, sym, seq):
        gid_base = gid.split('.')[0] if gid else ""
        if gid_base:
            if gid_base not in self.gene_to_sequence or len(seq) > len(self.gene_to_sequence[gid_base]):
                self.gene_to_sequence[gid_base] = seq
        if sym: self.symbol_to_sequence[sym.upper()] = seq
    
    def get_sequence(self, identifier: str) -> Optional[str]:
        seq = self.symbol_to_sequence.get(identifier.upper())
        if seq: return seq
        seq = self.gene_to_sequence.get(identifier.split('.')[0])
        if seq: return seq
        if self.gtf_parser:
            gid = self.gtf_parser.get_gene_id(identifier)
            if gid: return self.gene_to_sequence.get(gid)
        return None

class ArabidopsisBEDParser:
    def __init__(self, chrom_mapping: Dict[str, str] = None):
        self.chrom_mapping = chrom_mapping or {}
    
    def parse_file(self, bed_file: str) -> List[BindingPeak]:
        peaks = []
        opener = gzip.open if bed_file.endswith('.gz') else open
        mode = 'rt' if bed_file.endswith('.gz') else 'r'
        with opener(bed_file, mode) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'): continue
                p = self._parse_line(line)
                if p: peaks.append(p)
        logger.info(f"Parsed {len(peaks)} peaks from {bed_file}")
        return peaks
    
    def _parse_line(self, line: str) -> Optional[BindingPeak]:
        parts = line.split('\t')
        if len(parts) < 10: parts = line.split()
        if len(parts) < 10: return None
        chrom = parts[0].strip()
        if self.chrom_mapping:
            chrom = self.chrom_mapping.get(chrom)
            if not chrom: return None
        try: score = float(parts[9])
        except: score = 0.0
        strand = parts[4].strip()
        if strand not in ['+', '-']: strand = '+'
        return BindingPeak(chrom=chrom, start=int(parts[1]), end=int(parts[2]), peak_id=parts[3],
                           strand=strand, rbp_name=parts[5], method=parts[6], tissue=parts[7],
                           accession=parts[8], score=score)

class ArabidopsisProcessor:
    def __init__(self, genome_fasta, protein_fasta, gtf_file, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gtf_parser = GTFParser(gtf_file)
        self.protein_manager = ProteinSequenceManager(protein_fasta, self.gtf_parser)
        self.genome = pysam.FastaFile(genome_fasta)
        self.chrom_mapping = self._create_chrom_mapping()
        self.bed_parser = ArabidopsisBEDParser(self.chrom_mapping)
        self.stats = {'total_peaks': 0, 'valid_samples': 0, 'rbps_found': [], 'rbps_missing': []}
    
    def _create_chrom_mapping(self) -> Dict[str, str]:
        mapping = {}
        for chrom in self.genome.references:
            base = chrom.split()[0]
            mapping[base] = base
            mapping[f"Chr{base}"] = base
            mapping[f"chr{base}"] = base
        return mapping
    
    def extract_sequence(self, chrom, start, end, strand='+') -> Optional[str]:
        try:
            seq = self.genome.fetch(chrom, start, end).upper().replace('T', 'U')
            if strand == '-':
                comp = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
                seq = ''.join(comp.get(b, 'N') for b in seq[::-1])
            return seq
        except: return None
    
    def merge_peaks(self, peaks, distance=10):
        if not peaks: return []
        sorted_p = sorted(peaks, key=lambda x: (x.chrom, x.start))
        merged = []
        curr = {'chrom': sorted_p[0].chrom, 'start': sorted_p[0].start, 'end': sorted_p[0].end, 
                'strand': sorted_p[0].strand, 'scores': [sorted_p[0].score], 'pos': set(range(sorted_p[0].start, sorted_p[0].end))}
        for p in sorted_p[1:]:
            if p.chrom == curr['chrom'] and p.strand == curr['strand'] and p.start <= curr['end'] + distance:
                curr['end'] = max(curr['end'], p.end)
                curr['scores'].append(p.score)
                curr['pos'].update(range(p.start, p.end))
            else:
                merged.append((curr['chrom'], curr['start'], curr['end'], curr['strand'], np.mean(curr['scores']), curr['pos']))
                curr = {'chrom': p.chrom, 'start': p.start, 'end': p.end, 'strand': p.strand, 'scores': [p.score], 'pos': set(range(p.start, p.end))}
        merged.append((curr['chrom'], curr['start'], curr['end'], curr['strand'], np.mean(curr['scores']), curr['pos']))
        return merged

    def create_sample(self, chrom, center, strand, rbp, gid, pseq, bpos, wsize, score, method, tissue):
        start, end = center - wsize//2, center + wsize//2 + 1
        if start < 0 or end > self.genome.get_reference_length(chrom): return None
        rna = self.extract_sequence(chrom, start, end, strand)
        if not rna or len(rna) != wsize or rna.count('N')/len(rna) > 0.1: return None
        labels = [0]*wsize
        for p in bpos:
            rel = p - start if strand == '+' else end - 1 - p
            if 0 <= rel < wsize: labels[rel] = 1
        if sum(labels) == 0: return None
        return ArabidopsisBindingSample(rna, pseq, labels, f"arab_{rbp}_{chrom}_{start}_{strand}", rbp_name=rbp, gene_id=gid, chrom=chrom, start=start, end=end, strand=strand, method=method, tissue=tissue, score=score)

    def process_rbp(self, rbp, peaks, pseq, gid, wsize=101, stride=50):
        samples = []
        grouped = defaultdict(list)
        for p in peaks: grouped[(p.chrom, p.strand)].append(p)
        for (chrom, strand), group in grouped.items():
            merged = self.merge_peaks(group)
            for _, rs, re, rst, sc, bp in merged:
                if (re-rs) <= wsize:
                    s = self.create_sample(chrom, (rs+re)//2, rst, rbp, gid, pseq, bp, wsize, sc, group[0].method, group[0].tissue)
                    if s: samples.append(s)
                else:
                    for c in range(rs + wsize//2, re - wsize//2, stride):
                        s = self.create_sample(chrom, c, rst, rbp, gid, pseq, bp, wsize, sc, group[0].method, group[0].tissue)
                        if s: samples.append(s)
        return samples

    def process(self, bed_file, wsize=101, neg_ratio=1.0):
        peaks = self.bed_parser.parse_file(bed_file)
        self.stats['total_peaks'] = len(peaks)
        rbp_groups = defaultdict(list)
        for p in peaks: rbp_groups[p.rbp_name].append(p)
        all_samples = []
        for rbp, pks in rbp_groups.items():
            pseq = self.protein_manager.get_sequence(rbp)
            gid = self.gtf_parser.get_gene_id(rbp) or rbp
            if not pseq:
                self.stats['rbps_missing'].append(rbp)
                continue
            samples = self.process_rbp(rbp, pks, pseq, gid, wsize)
            all_samples.extend(samples)
            self.stats['rbps_found'].append(rbp)
        if neg_ratio > 0 and all_samples:
            all_samples.extend(self.generate_negatives(all_samples, int(len(all_samples)*neg_ratio), wsize))
        return all_samples

    def generate_negatives(self, positives, n_neg, wsize):
        logger.info(f"Generating {n_neg} negative samples")
        negatives = []
        pos_regions = defaultdict(list)
        for s in positives: pos_regions[s.chrom].append((s.start, s.end))
        rbp_info = {s.rbp_name: (s.protein_sequence, s.gene_id) for s in positives}
        rbp_list, chrom_list = list(rbp_info.keys()), list(pos_regions.keys())
        attempts, max_att = 0, n_neg * 30
        while len(negatives) < n_neg and attempts < max_att:
            attempts += 1
            chrom, rbp = np.random.choice(chrom_list), np.random.choice(rbp_list)
            clen = self.genome.get_reference_length(chrom)
            center = np.random.randint(wsize, clen - wsize)
            start, end = center - wsize//2, center + wsize//2 + 1
            if any(not (end+500 < ps or start-500 > pe) for ps, pe in pos_regions[chrom]): continue
            strand = np.random.choice(['+', '-'])
            rna = self.extract_sequence(chrom, start, end, strand)
            if rna and len(rna) == wsize and rna.count('N')/wsize <= 0.1:
                pseq, gid = rbp_info[rbp]
                negatives.append(ArabidopsisBindingSample(rna, pseq, [0]*wsize, f"neg_{rbp}_{chrom}_{start}", source="negative", rbp_name=rbp, gene_id=gid, chrom=chrom, start=start, end=end, strand=strand))
        return negatives

    def save(self, samples, name="arabidopsis_unified.json"):
        path = self.output_dir / name
        data = {
            "metadata": {"source": "Arabidopsis CLIP-seq", "total": len(samples), "rbps": self.stats['rbps_found']},
            "samples": [s.__dict__ for s in samples]
        }
        with open(path, 'w') as f: json.dump(data, f, indent=2)
        logger.info(f"Saved to {path}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--genome', required=True)
    parser.add_argument('--proteins', required=True)
    parser.add_argument('--gtf', required=True)
    parser.add_argument('--bed', required=True)
    parser.add_argument('--output_dir', default='data/arabidopsis/processed')
    parser.add_argument('--window_size', type=int, default=101)
    parser.add_argument('--neg_ratio', type=float, default=1.0)
    args = parser.parse_args()
    
    proc = ArabidopsisProcessor(args.genome, args.proteins, args.gtf, args.output_dir)
    samples = proc.process(args.bed, args.window_size, args.neg_ratio)
    if samples: proc.save(samples)

if __name__ == "__main__":
    main()
