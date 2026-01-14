import torch
from torch.utils.data import Dataset
from utils import expand_cigar, VOCAB, qual_to_ids
import torch.nn.functional as F
import random

class CigarQualDataset(Dataset):
    def __init__(self, sam_path, chunk_size, stride, max_records=None, split='train', train_ratio=0.8, seed=42):
        self.chunks = []
        all_chunks = []
        
        with open(sam_path) as fh:
            for line in fh:
                if line.startswith("@"): 
                    continue
                parts = line.split("\t")
                if len(parts) < 11:
                    continue
                cigar, qual = parts[5], parts[10]
                ops = expand_cigar(cigar)
                qids = qual_to_ids(qual)

                seq_len = len(ops)
                if seq_len <= 10:
                    continue

                # break into overlapping chunks
                for start in range(0, seq_len, stride):
                    end = min(start + chunk_size, len(ops))
                    chunk_ops = ops[start:end]
                    chunk_q = qids[start:end]

                    chunk_len = min(len(chunk_ops), len(chunk_q))
                    chunk_ops = chunk_ops[:chunk_len]
                    chunk_q = chunk_q[:chunk_len]

                    if not chunk_ops:
                        continue
                    # add BOS/EOS tokens
                    ids = [VOCAB["<BOS>"]] + [VOCAB.get(o, VOCAB["X"]) for o in chunk_ops]
                    qids_chunk = [0] + chunk_q
                    is_last = end >= len(ids)
                    if is_last:
                        ids.append(VOCAB["<EOS>"])
                        qids_chunk.append(0)
                    all_chunks.append((ids, qids_chunk))
                
                if max_records and len(all_chunks) >= max_records:
                        break
        
        # Split data into train/eval
        random.seed(seed)
        random.shuffle(all_chunks)
        
        split_idx = int(len(all_chunks) * train_ratio)
        if split == 'train':
            self.chunks = all_chunks[:split_idx]
        elif split == 'eval':
            self.chunks = all_chunks[split_idx:]
        else:
            raise ValueError("split must be 'train' or 'eval'")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return torch.tensor(self.chunks[idx][0]), torch.tensor(self.chunks[idx][1])
    
def collate_fn(batch):
    ops_batch, qual_batch = zip(*batch)
    ops_batch = torch.nn.utils.rnn.pad_sequence(ops_batch, batch_first=True, padding_value=VOCAB["<PAD>"])
    qual_batch = torch.nn.utils.rnn.pad_sequence(qual_batch, batch_first=True, padding_value=0)

    return ops_batch, qual_batch