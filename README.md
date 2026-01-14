Transformer-Based DNA Sequencing Read Generator

This project implements a transformer-based model for generating realistic synthetic DNA sequencing reads, designed to simulate data produced by modern sequencing technologies.
The tool models read length distributions, base quality (Phred scores), and sequencing errors and supports Illumina, Pacific Biosciences (PacBio), and Oxford Nanopore Technologies (ONT).

The project was developed as part of a diploma thesis in bioinformatics and compares a statistical (rule-based) simulator with a deep learning transformer model trained on real sequencing data.

Motivation

Sequencing devices are expensive, and real sequencing data is often:

limited in quantity,

restricted due to ethical or legal reasons,

technology-specific and hard to generalize.

Synthetic read generators enable:

development and testing of genome assembly and alignment algorithms,

controlled benchmarking of bioinformatics tools,

data augmentation for machine learning models.

While traditional simulators rely on fixed statistical distributions, this project explores whether transformers can learn complex sequencing patterns directly from data.

Features
Supported Sequencing Technologies

Illumina (MiSeq) – short, high-accuracy reads

PacBio (SMRT RS II) – long reads with higher error rates

Oxford Nanopore (MinION) – ultra-long reads with complex error patterns

Two Generation Approaches

Rule-based generator

Uses statistical distributions for:

read length,

base quality,

substitution, insertion, and deletion errors

Transformer-based generator

Learns sequencing behavior directly from aligned reads

Generates:

CIGAR operation sequences (=, X, I, D)

per-base Phred quality scores

Converts generated operations into FASTQ reads using a reference genome

Transformer Model Overview

Architecture: Transformer encoder

Input tokens:

Expanded CIGAR operations

Discretized Phred quality scores

Special tokens: <BOS>, <EOS>, <PAD>

Dual output heads:

CIGAR operation prediction

Quality score prediction

Training:

Autoregressive next-token prediction

Weighted loss to handle rare EOS tokens

Fragmented long reads (up to 1024 tokens)

The model learns joint dependencies between sequencing errors and base quality, producing more realistic synthetic reads than purely statistical methods.

Input & Output Formats
Input

FASTA – reference genome

SAM/BAM – aligned reads for training the transformer

Output

FASTQ – synthetic sequencing reads with Phred+33 quality encoding
