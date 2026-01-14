#this file contains linux comand line functions
import argparse

def defineArguments(parser):
    parser.add_argument("reference", type=str, help="FASTA datoteka s referencom")
    parser.add_argument("num_reads", type=int, help="Broj očitanja koja će se generirati")
    parser.add_argument("output_fastq", type=str, help="Izlazna FASTQ datoteka za generirana očitanja")
    parser.add_argument("--input_fastq", type=str, help="Ulazna FASTQ datoteka s očitanjima")
    parser.add_argument("--num_chim_reads", type=float, default=0.05, help="Postotak chimeric ocitanja")
    parser.add_argument("--substitution_rate", type=float, help="Postotak zamjena")
    parser.add_argument("--insertion_rate", type=float, help="Postotak umetanja")
    parser.add_argument("--deletion_rate", type=float, help="Postotak brisanja")
    parser.add_argument("--read_mean", type=int, help="Prosječna duljina očitanja")
    parser.add_argument("--read_stddev", type=int, help="Standardna devijacija duljine očitanja")
    parser.add_argument('--technology', default = 'PacBio', choices=['PacBio', 'Illumina', 'ONT'], help="Tehnologija korištena za dobivanje sekvenci: PacBio - Sequel II System - HiFi reads, ONT - PromethION with R10.4 Flow Cells")
    
    