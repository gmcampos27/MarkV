#!/bin/bash

echo "Starting BWA"

#Diretório arquivos entrada (fastq)

dir="/mnt/project2/svetoslav_slavov/metaviromaHumano/QC/Transplante_v3"
echo "Diretório de entrada: $dir"

#Diretório de saída Não Mapeado (bam)
dirout="/mnt/project2/svetoslav_slavov/metaviromaHumano/noHost/Unmapped/Transplante_v3"
echo "Diretório de saída: $dirout"

#Diretório do genoma de referência
ref="/mnt/project2/svetoslav_slavov/metaviromaHumano/genomes/Homo_sapiens/GRCh38_latest_genomic.fna"

for file in $dir/*_R1_trimmed.fastq.gz; do 
    if [ -f "$file" ]; then
        echo "Arquivo de entrada: $file"
        name=$(basename $file)
        name=${name%_R1_trimmed.fastq.gz}
        echo "Nome do arquivo: $name"
        echo "Executando BWA"
        bwa mem -t 16 -P  $ref $file ${file/_R1_trimmed.fastq.gz/_R2_trimmed.fastq.gz} | samtools view -b -f 4 > $dirout/${name}.bam
        samtools fastq -1 $dirout/${name}_unmapped_R1.fastq.gz -2 $dirout/${name}_unmapped_R2.fastq.gz $dirout/${name}.bam
        echo "BWA finalizado para $file"
    fi
done    
