import re, random
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


VOCAB = {"<PAD>":0,"<BOS>":1,"<EOS>":2,"=":3,"X":4,"I":5,"D":6, "S":7}
INV_VOCAB = {v:k for k,v in VOCAB.items()}

REF_CONSUMING = {"=","X","D"}
READ_CONSUMING = {"=", "X", "I", "S"}

_cigar_pat = re.compile(r'(\d+)([MIDNSHP=X])')

def parse_cigar_ops(cigar: str):
    return [(int(cnt), op) for cnt, op in _cigar_pat.findall(cigar)]

def expand_cigar(cigar: str):
    ops=[]
    for cnt,op in parse_cigar_ops(cigar):
        if op in ('H','P'): continue
        ops.extend([op]*cnt)
    return ops

def compress_ops(ops):
    if not ops: return "*"
    parts,cur,cnt=[],ops[0],1
    for o in ops[1:]:
        if o==cur: cnt+=1
        else: parts.append(f"{cnt}{cur}"); cur,cnt=o,1
    parts.append(f"{cnt}{cur}")
    return ''.join(parts)

def qual_to_ids(qual_str, max_q=40):
    return [min(ord(ch)-33, max_q) for ch in qual_str]

def ids_to_qual(ids):
    return ''.join(chr(q+33) for q in ids)

# -------------------
# FASTA parsing
# -------------------
def read_fasta(path):
    seqs={}; name=None; cur=[]
    with open(path) as fh:
        for line in fh:
            line=line.strip()
            if not line: continue
            if line.startswith(">"):
                if name: seqs[name]=''.join(cur).upper()
                name=line[1:].split()[0]; cur=[]
            else: cur.append(line)
        if name: seqs[name]=''.join(cur).upper()
    return seqs

# -------------------
# Read synthesis
# -------------------
def synthesize_read_from_ops(reference, chrom, start, ops, qual_ids):
    ref=reference[chrom]
    rpos=start
    read=[]
    qual=[]
    q_iter=iter(qual_ids)
    for op in ops:
        if op=="=":
            base = ref[rpos] if rpos<len(ref) else random.choice("ACGT")
            read.append(base); qual.append(next(q_iter,30)); rpos+=1
        elif op=="X":
            if rpos<len(ref):
                choices=[b for b in "ACGT" if b!=ref[rpos]]
                base=random.choice(choices)
            else:
                base=random.choice("ACGT")
            read.append(base); qual.append(next(q_iter,30)); rpos+=1
        elif op=="I":
            read.append(random.choice("ACGT")); qual.append(next(q_iter,30))
        elif op=="S":
            read.append(random.choice("ACGT")); qual.append(next(q_iter,30))
        elif op=="D":
            rpos+=1
    return ''.join(read), ids_to_qual(qual), rpos


def write_to_fastq(reads, quals, path):
    with open(path, "w") as fh:
        for i in range(len(reads)):
            fh.write(f"@read{i+1}\n")
            fh.write(f"{reads[i]}\n")
            fh.write("+\n")
            fh.write(f"{quals[i]}\n")
    return


def plot_embeddings(embeddings):
    emb_3d = TSNE(n_components=2, perplexity=3).fit_transform(embeddings)

    # Plot
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot()
    ax.scatter(emb_3d[:,0], emb_3d[:,1], c='blue')

    # Optionally annotate first few tokens
    for i, token in enumerate(VOCAB):
        ax.text(emb_3d[i,0], emb_3d[i,1], token)

    plt.show(block=True)