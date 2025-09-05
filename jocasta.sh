#!/bin/bash

# Loop em todos os diretórios do SPAdes
# Assumindo que os diretórios seguem o padrão "spades_output_NF*"
echo "Iniciando processamento dos diretórios SPAdes..."
for dir in /mnt/project2/svetoslav_slavov/metaviromaHumano/spades/spades_output_NF*/; do
    echo "Processando $dir"
    
    fasta="$dir/scaffolds.fasta"
    
    # Verifica se o arquivo existe
    if [ ! -f "$fasta" ]; then
        echo "X Arquivo não encontrado: $fasta"
        continue
    fi

    # Etapa 1: extrair tamanhos
    grep '>' "$fasta" | cut -f 4 -d '_' > "$dir/scaffolds_length.txt"

    # Etapa 2: extrair IDs
    grep '>' "$fasta" > "$dir/scaffolds_ids.txt"

    # Etapa 3: unir, filtrar e gerar lista final
    paste "$dir/scaffolds_ids.txt" "$dir/scaffolds_length.txt" | \
        awk '$2 > 500' | cut -f 1 | sed 's/>//g' > "$dir/scaffolds_ids_above_500.txt"

    # Etapa 4: gerar novo fasta
    seqtk subseq "$fasta" "$dir/scaffolds_ids_above_500.txt" > "$dir/scaffolds_above_500.fasta"

    echo "✅ Finalizado: $dir"
done

# 4a etapa: fazer blast dos scaffolds contra o banco nt
#sbatch -n 16 --job-name "BLAST" --wrap 'blastn -query scaffolds_above_500.fasta -db /database/ncbi-blast+/nt -outfmt "6 qseqid sallseqid pident length qcovs qstart qend sstart send evalue bitscore staxids" -out scaffolds_above_500_vs_nt.tsv -num_threads 16'
#https://www.metagenomics.wiki/tools/blast/blastn-output-format-6

# 5a etapa: filtrar por query coverage acima de 70%
# cat scaffolds_above_500_vs_nt.tsv | awk '$5 > 70' > scaffolds_above_500_vs_nt_qcov70.tsv

# 6a etapa: filtrar resultados por ids "únicos" de scaffolds e armazená-los em um arquivo
#sort -r -n -k 5 scaffolds_above_500_vs_nt_qcov70.tsv | awk '!_[$1]++' > scaffolds_above_500_vs_nt_qcov70_uniq.tsv
#cut -f 12 scaffolds_above_500_vs_nt_qcov70_uniq.tsv > taxids.txt
