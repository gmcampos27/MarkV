#!/bin/bash

echo "Extração Matéria Escura - Kraken2 e Bracken"
echo "Olá, eu sou o Wall-e e estou aqui para ajudar você!"

dir="noHost/Unmapped/"
echo "Diretório de entrada: $dir"

dirout="metadata/"
echo "Diretório de saída: $dirout"

dirdb="database/kraken2.2/"

for FILE in ${dir}*.R1.fastq.gz; do
    if [ -f "$FILE" ]; then
        echo "Arquivo de entrada: $FILE"
        name=$(basename "$FILE")
        name=${name%.R1.fastq.gz}
        
        echo "Rodando Kraken2 com:"
        echo "Arquivo 1: $dir/${name}.R1.fastq.gz"
        echo "Arquivo 2: $dir/${name}.R2.fastq.gz"
        echo "Saída: $dirout/${name}_kraken2.tsv"

        echo "Extraindo a matéria escura do arquivo: $name"
        kraken2 --db "$dirdb" --threads 24 --paired "$dir/${name}.R1.fastq.gz" "$dir/${name}.R2.fastq.gz" --unclassified-out "$dirout/${name}_unclassified#.fastq"
    fi 
done
