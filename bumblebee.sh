#!/bin/bash 

echo "RiboDetector started at $(date)"

#Diretório Arquivos de entrada
dirin="/noHost/Unmapped/Transplante/"
echo "Diretório de entrada: $dirin"

#Diretório Arquivos de saída
dirout="noHost/Unmapped/Transplante/NonrRNA/"
echo "Diretório de saída: $dirout"

for file in $dirin*_R1.fastq.gz; do
    if [ -f "$file" ]; then
        echo "Arquivo de entrada: $file"
        name=$(basename $file)
        name=${name%_unmapped_R1.fastq.gz}
        echo "Nome do arquivo: $name"
        ribodetector_cpu -t 20 -l 150 -i $file ${file/_unmapped_R1.fastq.gz/_unmapped_R2.fastq.gz} -e norrna -o $dirout/${name}_nonrna_R1.fastq.gz $dirout/${name}_nonrna_R2.fastq.gz --log $dirout/${name}_log.txt
    fi
done
