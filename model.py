import math, torch, torch.nn as nn
from utils import VOCAB, INV_VOCAB
import torch.nn.functional as F
from tqdm import tqdm

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=8192):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.register_buffer("pe", self._make_pe(max_len))

    def _make_pe(self, length):
        pe = torch.zeros(length, self.d_model)
        pos = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)  # shape [1, length, d_model]

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class CigarQualTransformer(nn.Module):
    def __init__(self, vocab_size, len_qual, d_model=64, nhead=8, ff=512, num_layers=6):
        super().__init__()
        self.emb_cigar = nn.Embedding(vocab_size, d_model, padding_idx=VOCAB["<PAD>"])
        self.emb_qual = nn.Embedding(len_qual, d_model, padding_idx=0)
        self.pe = PositionalEncoding(d_model)

        layer = nn.TransformerEncoderLayer(d_model, nhead,dim_feedforward=ff, activation="gelu", batch_first=True, dropout=0.1)

        self.tr = nn.TransformerEncoder(layer, num_layers)

        self.norm = nn.LayerNorm(d_model)
        self.head_ops = nn.Linear(d_model, vocab_size)
        self.head_qual = nn.Linear(d_model, len_qual)

    def forward(self, cigar_ids, qual_ids):
        x_cigar = self.emb_cigar(cigar_ids) * math.sqrt(self.emb_cigar.embedding_dim)
        x_qual  = self.emb_qual(qual_ids) * math.sqrt(self.emb_qual.embedding_dim)
        x = x_cigar + x_qual 
        x = self.pe(x)
        mask = (cigar_ids == VOCAB["<PAD>"])
        x = self.tr(x, src_key_padding_mask=mask)
        x = self.norm(x)
        return self.head_ops(x), self.head_qual(x)

def train(model, device, dataloader, optimizer, epochs=3, eval_dataloader=None):
    model.to(device)
    for ep in range(epochs):
        # Training phase
        model.train()
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {ep+1}/{epochs} - Training")  
        for cigar_ids, qual_ids in pbar:

            cigar_ids = cigar_ids.to(device)
            qual_ids = qual_ids.to(device)
            optimizer.zero_grad()
            logits_ops, logits_qual = model(cigar_ids, qual_ids)

            # Shift targets
            target_ops = cigar_ids[:, 1:].contiguous().view(-1)
            pred_ops = logits_ops[:, :-1].contiguous().view(-1, logits_ops.size(-1))
            #weights = torch.ones(len(VOCAB), device=device)
            #weights[VOCAB["<EOS>"]] = 2
            loss_ops = F.cross_entropy(pred_ops, target_ops, ignore_index=VOCAB["<PAD>"], label_smoothing=0.0)

            target_q = qual_ids[:, 1:].contiguous().view(-1)
            pred_q = logits_qual[:, :-1].contiguous().view(-1, logits_qual.size(-1))
            loss_q = F.cross_entropy(pred_q, target_q, ignore_index=0, label_smoothing=0.0)

            loss = loss_ops + 0.8 * loss_q
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}, {loss_ops.item():.4f}, {loss_q.item():.4f}") 

        avg_train_loss = total_loss / len(dataloader)
        
        # Evaluation phase
        if eval_dataloader is not None:
            print(f"Epoch {ep+1} - Train Loss: {avg_train_loss:.4f}")
            eval_loss, eval_loss_ops, eval_loss_qual = evaluate(model, device, eval_dataloader)
            print(f"Epoch {ep+1} - Eval Loss: {eval_loss:.4f} (Ops: {eval_loss_ops:.4f}, Qual: {eval_loss_qual:.4f})")
        else:
            print(f"Epoch {ep+1} avg loss: {avg_train_loss:.4f}")
        
        print("-" * 50)


@torch.no_grad()
def evaluate(model, device, dataloader):
    model.to(device)
    model.eval()
    total_loss = 0.0
    total_loss_ops = 0.0
    total_loss_qual = 0.0
    total_samples = 0
    
    pbar = tqdm(dataloader, desc="Evaluating")
    for cigar_ids, qual_ids in pbar:
        cigar_ids = cigar_ids.to(device)
        qual_ids = qual_ids.to(device)
        
        logits_ops, logits_qual = model(cigar_ids, qual_ids)

        # Shift targets (same as in training)
        target_ops = cigar_ids[:, 1:].contiguous().view(-1)
        pred_ops = logits_ops[:, :-1].contiguous().view(-1, logits_ops.size(-1))

        loss_ops = F.cross_entropy(pred_ops, target_ops, ignore_index=VOCAB["<PAD>"], reduction='sum')

        target_q = qual_ids[:, 1:].contiguous().view(-1)
        pred_q = logits_qual[:, :-1].contiguous().view(-1, logits_qual.size(-1))
        loss_q = F.cross_entropy(pred_q, target_q, ignore_index=0, reduction='sum')

        # Count non-padded tokens for proper averaging
        non_pad_mask = (target_ops != VOCAB["<PAD>"])
        non_zero_mask = (target_q != 0)
        
        total_loss_ops += loss_ops.item()
        total_loss_qual += loss_q.item()
        total_samples += non_pad_mask.sum().item()  # Use ops tokens for total count
        
        combined_loss = loss_ops +  loss_q
        total_loss += combined_loss.item()
        
        pbar.set_postfix(loss=f"{combined_loss.item()/(non_pad_mask.sum().item()):.4f}")

    avg_loss = total_loss / total_samples
    avg_loss_ops = total_loss_ops / total_samples
    avg_loss_qual = total_loss_qual / total_samples
    
    print(f"Eval Loss: {avg_loss:.4f} (Ops: {avg_loss_ops:.4f}, Qual: {avg_loss_qual:.4f})")
    return avg_loss, avg_loss_ops, avg_loss_qual


@torch.no_grad()
def generate_long_read(model, device, chunk_size, max_len=100000, temperature=1.0):
    model.to(device)
    model.eval()
    generated_ops = [VOCAB["<BOS>"]]
    generated_qual = [0]
    banned_tokens = [VOCAB["<BOS>"], VOCAB["<PAD>"]]
    
    while len(generated_ops) < max_len:

        context_ops = generated_ops[-chunk_size:]
        context_quals = generated_qual[-chunk_size:]
        context_ops_tensor = torch.tensor([context_ops], device=device)
        context_quals_tensor = torch.tensor([context_quals], device=device)
        
        logits_ops, logits_qual = model(context_ops_tensor, context_quals_tensor)

        current_logits_ops = logits_ops[0, -1] / temperature
        current_logits_qual = logits_qual[0, -1] / temperature
        
        current_logits_ops[banned_tokens] = -float('inf')

        next_op = torch.multinomial(F.softmax(current_logits_ops, dim=-1), 1).item()
        next_q = torch.multinomial(F.softmax(current_logits_qual, dim=-1), 1).item()

        if next_op == VOCAB["<EOS>"]:
            break

        generated_ops.append(next_op)
        generated_qual.append(next_q)
    # remove the first <BOS> but keep rest
    return generated_ops[1:], generated_qual[1:]