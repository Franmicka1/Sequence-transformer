import argparse, torch, random
from torch.utils.data import DataLoader
from dataset import CigarQualDataset, collate_fn
from model import CigarQualTransformer, train, evaluate, generate_long_read
from utils import INV_VOCAB, compress_ops, read_fasta, synthesize_read_from_ops, write_to_fastq, plot_embeddings
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--sam",required=True)
    ap.add_argument("--ref",required=True)
    ap.add_argument("--epochs",type=int,default=3)
    ap.add_argument("--batch",type=int,default=64)
    ap.add_argument("--chunk_size", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=1024)
    ap.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_train_records", type=int, default=200000, help="Limit number of SAM records to load (for faster prototyping).")
    ap.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of data to use for training (rest for eval)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for train/eval split")
    args=ap.parse_args()

    # Create train and eval datasets
    train_ds = CigarQualDataset(args.sam, max_records=args.max_train_records,
                               chunk_size=args.chunk_size, stride=args.stride,
                               split='train', train_ratio=args.train_ratio, seed=args.seed)
    
    eval_ds = CigarQualDataset(args.sam, max_records=args.max_train_records,
                              chunk_size=args.chunk_size, stride=args.stride,
                              split='eval', train_ratio=args.train_ratio, seed=args.seed)
    
    print(f"Train samples: {len(train_ds)}, Eval samples: {len(eval_ds)}")
    
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, pin_memory=True, collate_fn=collate_fn)
    eval_dl = DataLoader(eval_ds, batch_size=args.batch, shuffle=False, pin_memory=True, collate_fn=collate_fn)

    model=CigarQualTransformer(vocab_size=len(INV_VOCAB),len_qual=41)
    optim=torch.optim.AdamW(model.parameters(),lr=1e-3)

    model_params_file = "model_params_illumina.pth"
    
    # Training
    train(model,torch.device(args.device),train_dl,optim,epochs=args.epochs)
    torch.save(model.state_dict(), model_params_file)

    # Evaluation
    print("\n--- Evaluation ---")
    model.load_state_dict(torch.load(model_params_file))
    evaluate(model, torch.device(args.device), eval_dl)
    
    #embeddings = model.emb_cigar.weight.detach().cpu().numpy()
    #plot_embeddings(embeddings)

    # Generation (same as before)
    reads_lst, quals_lst = [], []
    for i in range(100):
        print(f"\n--- Generating read {i+1} ---")
        ops_ids,quals=generate_long_read(model,torch.device(args.device), chunk_size=args.chunk_size)

        ops=[INV_VOCAB[o] for o in ops_ids]
        cigar=compress_ops(ops)
        print("Generated CIGAR:",cigar)

        refs=read_fasta(args.ref)
        chrom=random.choice(list(refs.keys()))
        start=random.randint(0,max(0,len(refs[chrom])-len(ops)))
        read_seq,qual_str,end=synthesize_read_from_ops(refs,chrom,start,ops,quals)
        reads_lst.append(read_seq)
        quals_lst.append(qual_str)

    write_to_fastq(reads_lst, quals_lst, "synthetic_read_Illumina.fastq")
    print("\n--- Synthetic alignment ---")
    print(f"Reference: {chrom}")
    print(f"Start pos: {start+1}")
    print(f"CIGAR: {cigar}")
    print(f"Read: {read_seq}")
    print(f"Qual: {qual_str}")

if __name__=="__main__": 
    main()