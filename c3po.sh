#!/bin/bash

echo "Montagem de contigs"

dir="/mnt/project2/svetoslav_slavov/metaviromaHumano/metadata/NF_Amostras/"
echo "Diretório de entrada: $dir"

dirout="/mnt/project2/svetoslav_slavov/metaviromaHumano/noHost/Paired/"
echo "Diretório de saída: $dirout"

echo executando o trimmomatic para montagem de contigs

for FILE in ${dir}*_unclassified_1.fastq; do
    if [ -f "$FILE" ]; then
        echo "Arquivo de entrada: $FILE"
        name=$(basename "$FILE")
        name=${name%_1.fastq}  # Remove "_1.fastq", restando "NF1_unclassified"

        echo "Rodando Trimmomatic com:"
        echo "Arquivo 1: $dir/${name}_1.fastq"
        echo "Arquivo 2: $dir/${name}_2.fastq"
        echo "Saída: $dirout/${name}_contigs.fasta"

        echo "Montando contigs do arquivo: $name"
        #trimmomatic PE -phred33 "$dir/${name}_1.fastq" "$dir/${name}_2.fastq" "$dirout/${name}_R1_paired.fq.gz" "$dirout/${name}_unpaired_R1.fq.gz" "$dirout/${name}_R2_paired.fq.gz" "$dirout/${name}_unpaired_R2.fq.gz" -threads 10 SLIDINGWINDOW:4:15
    fi 
done

dir_pair="/mnt/project2/svetoslav_slavov/metaviromaHumano/noHost/Paired/"
dir_unpair="/mnt/project2/svetoslav_slavov/metaviromaHumano/noHost/Unpaired/"
dir_spades="/mnt/project2/svetoslav_slavov/metaviromaHumano/spades/"

echo "Executando SPAdes para montagem de contigs"

for FILE in ${dir_pair}*_unclassified_R1_paired.fq.gz; do
    if [ -f "$FILE" ]; then
        echo "Arquivo de entrada: $FILE"
        name=$(basename "$FILE")
        name=${name%_unclassified_R1_paired.fq.gz}  # Remove "_unclassified_R1_paired.fq.gz", restando "NF1"
        echo "Rodando SPAdes com:"
        echo "Arquivo 1: $dir_pair/${name}_unclassified_R1_paired.fq.gz"
        echo "Arquivo 2: $dir_pair/${name}_unclassified_R2_paired.fq.gz"
        echo "Arquivo não pareado: $dir_unpair/${name}_unpaired.fq.gz"
        echo "Saída: $dir_spades/spades_output_${name}/"

        echo "Montando contigs do arquivo: $name"
        spades.py --meta -1 "$dir_pair/${name}_unclassified_R1_paired.fq.gz" -2 "$dir_pair/${name}_unclassified_R2_paired.fq.gz" -s "$dir_unpair/${name}_unpaired.fq.gz" -k 21,33,55,77 -t 16 -o "$dir_spades/spades_output_${name}/"
    fi
done