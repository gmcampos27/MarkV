# -*- coding: utf-8 -*-
import numpy as np
import os
import pandas as pd
from Bio import SeqIO
import csv
from Bio.Seq import Seq
from Bio.SeqUtils import gc_fraction

os.chdir("/path") #Diretório

new_fasta = "processed/Familias.fasta"
gc_data = []

# Calcular o conteúdo de GC
for record in SeqIO.parse(new_fasta, "fasta"):
    gc_content = gc_fraction(record.seq) * 100  # Calcular GC e converter para %
    gc = round(gc_content, 2)  # Arredondar
    gc_data.append({"Accession": record.id, "GC_Content": gc})

# Criar um DataFrame
gc_df = pd.DataFrame(gc_data)

gc_df.to_csv("gc_content.csv", index=False)

filtered_metadata = pd.read_csv("sheets/Family_Metadata.csv")
gc_content = pd.read_csv("gc_content.csv")

# Mesclar os dois DataFrames usando 'Accession' como chave
merged_data = pd.merge(filtered_metadata, gc_content, on="Accession")

merged_data.to_csv("Meta_GC.csv", index=False)

def calc_tetranucleotide_frequency(sequence):
    # Gerar tetranucleotídeos
    tetranucleotides = [sequence[i:i+4] for i in range(len(sequence) - 3)]
    # Filtrar apenas os válidos
    valid_tetranucleotides = [tet for tet in tetranucleotides if set(tet).issubset({"A", "C", "G", "T"})]
    # Contar as frequências
    counts = Counter(valid_tetranucleotides)
    # Normalizar as frequências
    total = sum(counts.values())
    if total > 0:
        normalized_counts = {k: v / total for k, v in counts.items()}
    else:
        normalized_counts = 0
    return normalized_counts

from collections import Counter
FTN = []
for record in SeqIO.parse(new_fasta, "fasta"):
    freq_dict = calc_tetranucleotide_frequency(record.seq)
    freq_dict["Accession"] = record.id  # Adicionar o Accession como chave
    FTN.append(freq_dict)

meta_gc = pd.read_csv("Meta_GC.csv")
tetranuc_df = pd.DataFrame(FTN)
merged_df = pd.merge(meta_gc, tetranuc_df, on="Accession")

merged_df=merged_df.drop_duplicates()
merged_df.to_csv('sheets/Meta_GC_FTN.csv')
print("Features referentes a Frequencia de Tetranucleotídeo e GC% extraídas")
