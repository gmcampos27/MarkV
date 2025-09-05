#!/bin/bash

echo "Começando fastp"

# Diretório contendo os arquivos de entrada
input_dir="/mnt/project2/svetoslav_slavov/metaviromaHumano/rawData/Transplante_v3"
echo "Diretório de entrada: $input_dir"

# Diretório para salvar os arquivos de saída
output_dir="/mnt/project2/svetoslav_slavov/metaviromaHumano/QC/Transplante_v3"
echo "Diretório de saída: $output_dir"

# Loop para processar cada par de arquivos de entrada
for file1 in "$input_dir"/*R1_001.fastq.gz; do
    # Verifica se o arquivo R1 existe
    if [ -e "$file1" ]; then
        # Extrai o nome base do arquivo R1
        filename_base=$(basename "$file1" _R1_001.fastq.gz)
        
        # Verifica se o arquivo R2 correspondente existe
        file2="$input_dir"/"${filename_base}_R2_001.fastq.gz"
        if [ -e "$file2" ]; then
            # Executa o Fastp para o par de arquivos R1 e R2
            fastp -w 12 -i "$file1" -I "$file2" -o "$output_dir"/"$filename_base"_R1_trimmed.fastq.gz -O "$output_dir"/"$filename_base"_R2_trimmed.fastq.gz -q 30 -g -x -c -D --dup_calc_accuracy 1 -h "$output_dir"/"$filename_base".html
            
        else
            echo "Arquivo $file2 não encontrado para $file1."
        fi
    else
        echo "Arquivo $file1 não encontrado."
    fi
done
