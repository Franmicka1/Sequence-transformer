import random
import argparse
import cmd_helper
import subprocess
import numpy as np
import json
from scipy.stats import skewnorm

class ReadSimulator:
    def __init__(self, args):
        self.reference = args.reference
        self.num_reads = args.num_reads
        self.output_fastq = args.output_fastq
        self.technology = args.technology
        self.num_chim_reads = args.num_chim_reads
        self.initialize_technology_args()
        self.override_techonlogy_args(args)
    
    def initialize_technology_args(self):
        file_name = 'technologies.json'
        with open(file_name, 'r') as file:
            json_data = json.load(file)
        for key,value in json_data['technologies'][self.technology].items():
            setattr(self, key, value)
                
    
    def override_techonlogy_args(self, args):
        if args.substitution_rate:
            self.error_profile['substitution_rate'] = args.substitution_rate
        if args.insertion_rate:
            self.error_profile['insertion_rate'] = args.insertion_rate
        if args.deletion_rate:
            self.error_profile['deletion_rate'] = args.deletion_rate
        if args.read_mean:
            self.read_mean = args.read_mean
        if args.read_stddev:
            self.read_stddev = args.read_stddev
        
    def load_reference(self):
        sequences = {}
        with open(self.reference, 'r') as f:
            current_seq = ""
            current_chrom = None
            for line in f:
                if line.startswith(">"):
                    if current_chrom:
                        sequences[current_chrom] = current_seq
                    current_chrom = line[1:].strip()
                    current_seq = ""
                else:
                    current_seq += line.strip()
            if current_chrom:
                sequences[current_chrom] = current_seq
        return sequences

    def get_sample(self, distribution):
        distribution_functions = {
            'normal': np.random.normal,
            'lognormal': np.random.lognormal,
            'exponential': np.random.exponential,
            'fixed': None
        }
        if distribution == 'normal':
            return int(distribution_functions['normal'](self.read_mean, self.read_stddev))
        elif distribution == 'lognormal':
            return int(distribution_functions['lognormal'](self.read_mean, self.read_stddev))   
        elif distribution == 'exponential':
            return int(distribution_functions['exponential'](self.read_mean))
        elif distribution == 'fixed':
            return self.read_length
        else:
            return max(1, int(np.random.normal(1000, 100)))

    def resolve_chim_reads(self, reads):
        while len(reads) > self.num_reads:
            combine_indexes = random.sample(range(len(reads)), 2)
            combine_indexes.sort(reverse=True)
            read0 = reads.pop(combine_indexes[0])
            read1 = reads.pop(combine_indexes[1])
            new_read = (read0 + read1)
            reads.append(new_read)

    def reverse_complement(self, read):
        complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
        reverse_complement_chance = 0.5
        if (random.random() > reverse_complement_chance):
            return ''.join(complement[base] for base in reversed(read))

    def introduce_errors(self, seq):
        error_seq = []
        for base in seq:
            r = random.random()
            if r < self.error_profile['substitution']:
                bases = ['A', 'T', 'C', 'G']
                bases.remove(base)
                error_seq.append(random.choice(bases))

            elif r < self.error_profile['substitution'] + self.error_profile['insertion']:
                error_seq.append(base)
                error_seq.append(random.choice(['A', 'T', 'C', 'G']))

            elif r < self.error_profile['substitution'] + self.error_profile['insertion'] + self.error_profile['deletion']:
                continue
            else:
                error_seq.append(base)
        return "".join(error_seq)

    def generate_illumina_reads(self, sequences):
        reads = []
        for _ in range(self.num_reads):
            chrom = random.choice(list(sequences.keys()))
            seq = sequences[chrom]

            if random.random() < 0.9:
                read_length = min(250, len(seq))
            else:
                read_length = random.randint(1, min(249, len(seq)))
            start = random.randint(0, len(seq) - read_length)
            read = seq[start:start + read_length]
            self.reverse_complement(read)
            reads.append(read)
        for index, read in enumerate(reads):
            read_with_errors = self.introduce_errors(read)
            if len(read_with_errors) > 251:
                    read_with_errors = read_with_errors[:251]
            reads[index] = read_with_errors
        return reads

    def generate_reads(self, sequences):
        if self.technology == "Illumina":
            return self.generate_illumina_reads(sequences)
        reads = []
        total_reads = int(self.num_reads * (1+self.num_chim_reads))
        for _ in range(total_reads):
            chrom = random.choice(list(sequences.keys()))
            seq = sequences[chrom]
            read_length = self.get_sample(self.read_length_distribution)
            start = random.randint(0, len(seq) - read_length)
            read = seq[start:start + read_length]
            self.reverse_complement(read)
            reads.append(read)
        self.resolve_chim_reads(reads)
        for index, read in enumerate(reads):
            read_with_errors = self.introduce_errors(read)
            reads[index] = read_with_errors

        return reads
    
    def initialize_quality_distributions(self, technology, reads):
        qual_mean, qual_stddev = [],[]
        longest_read = max(reads, key=len)
        if technology == "Illumina":
            length = self.read_length
            for i in range(length):
                if (i < 20):
                    frac = (20 - i) / (length - 20)
                    noise = 5*-(np.sqrt(frac))
                    qual_mean.append(self.quality_mean + noise)
                    noise = np.random.uniform(-0.5,0.5)
                    qual_stddev.append((self.quality_stddev + noise) * (1 + frac))
                elif i < 180: 
                    noise = np.random.normal(1,0.05)
                    qual_mean.append(self.quality_mean + noise)
                    noise = np.random.uniform(-0.5,0.5)
                    qual_stddev.append(self.quality_stddev + noise)
                else:
                    frac = (i - 180) / (length - 180) 
                    noise = np.random.normal(1,0.05) + 3*-(np.power(frac,2))
                    qual_mean.append(self.quality_mean + noise)
                    noise = np.random.uniform(-1,1)
                    qual_stddev.append((self.quality_stddev + noise)*(1+ frac))
        else:
            length = longest_read
            for i in range(len(length)):
                noise = np.random.normal(0,0.5)
                qual_mean.append(self.quality_mean + noise)
                noise = np.random.uniform(-1,1)
                qual_stddev.append(self.quality_stddev + noise)
        return qual_mean, qual_stddev
    
    def generate_illumina_quality(self, length, qual_means, qual_devs):
        qualities = np.ones(length) * qual_means[0:length]

        for i in range(length):
            noise = np.random.exponential(scale=qual_devs[i]) 
            qualities[i] -= noise

        qualities = np.clip(qualities, 0, 40)
        return ''.join(chr(int(q) + 33) for q in qualities)

    def generate_quality(self, length, quality_means, quality_stds):
        if self.technology == "Illumina":
            return self.generate_illumina_quality(length, quality_means, quality_stds)
        

        quality_means = np.array(quality_means[0:length])
        quality_stds = np.array(quality_stds[0:length])

        noise = np.random.exponential(scale=1)
        quality_means = [x - noise for x in quality_means]
        Q = np.random.normal(quality_means, quality_stds)
        Q = np.clip(Q, 0, 40)  # Ensure quality scores are within valid range
        return ''.join(chr(int(q) + 33) for q in Q) 


    def save_reads(self, reads):
        with open(self.output_fastq, 'w') as f:
            quality_means, quality_stds = self.initialize_quality_distributions(self.technology, reads)
            for i, read in enumerate(reads):
                read_id = f"read{i}"
                quality = self.generate_quality(len(read), quality_means, quality_stds)
                f.write(f"@{read_id} length={len(read)}\n")
                f.write(f"{read}\n")
                f.write("+\n")
                f.write(f"{quality}\n")
        print(f"Generirano {self.num_reads} očitanja u {self.output_fastq}.")

def main():
    parser = argparse.ArgumentParser(description="Simulacija sekvenciranja s pogreškama i testiranje pomoću Minimap2")
    cmd_helper.defineArguments(parser)
    args = parser.parse_args()
    simulator = ReadSimulator(args)
    sequences = simulator.load_reference()
    reads = simulator.generate_reads(sequences)
    simulator.save_reads(reads)
  
if __name__ == "__main__":
    main()