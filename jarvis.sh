#!/bin/bash

# Script para anotar scaffolds acima de 500bp usando HMMER com VFam e Pfam

echo "Iniciando anotação dos scaffolds acima de 500bp..."

# Caminhos para os bancos
vfam_db="database/vFam-A_2014.hmm"
pfam_db="database/Pfam-A.hmm"

# Output centralizado para domtblout
mkdir -p hmmer_vfam hmmer_pfam

# Loop em cada diretório
for dir in spades/spades_output*/; do
    echo "🔄 Processando $dir"

    fasta="$dir/scaffolds_above_500.fasta"

    if [ ! -f "$fasta" ]; then
        echo "❌ Arquivo não encontrado: $fasta"
        continue
    fi

    # 1. Tradução para proteínas
    pep="$dir/scaffolds_above_500_pep.fasta"
    transeq -sequence "$fasta" -outseq "$pep" -clean -frame 6

    # 2. Remover _1 do final dos IDs (evita erro no hmmsearch)
    pep_clean="$dir/scaffolds_above_500_pep_clean.fasta"
    sed 's/_1$//' "$pep" > "$pep_clean"

    # 3. HMMSEARCH VFam
    hmmsearch --cpu 16 --domtblout "${dir%/}_vfam_hits.txt" \
      --noali -E 0.00001 "$vfam_db" "$pep_clean" > "$dir/scaffolds_vs_vfam.out"

    # 4. HMMSEARCH Pfam
    hmmsearch --cpu 16 --domtblout "${dir%/}_pfam_hits.txt" \
      --noali -E 0.00001 "$pfam_db" "$pep_clean" > "$dir/scaffolds_vs_pfam.out"

    echo "✅ Anotação finalizada para $dir"
done
