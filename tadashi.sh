#!/bin/bash

set -euxo pipefail

echo "Olá, eu sou o Tadashi!"
echo "Programa inciado em $(date)"
echo "Prepare-se para a análise do metaviroma!"

# ========= CONFIG =========
dirdb="/storage/zuleika/volume3/database/kraken2.2/"  
THREADS=24
dir="/storage/vital/volume2/svetoslav_slavov/metaviromaHumano/noHost/Unmapped/Transplante_v3/NonrRNA/"
dirout="/storage/vital/volume2/svetoslav_slavov/metaviromaHumano/metadata/Transplante_v3/"
echo "Arquivos finais em: $dirout"
# ==========================

for file in $dir*_nonrna_R1.fastq.gz; do
    if [ -f "$file" ]; then
        echo "Arquivo de entrada: $file"
        name=$(basename $file)
        name=${name%_nonrna_R1.fastq.gz}
        echo "Nome do arquivo: $name"

        # Classificação taxonômica com Kraken2
        kraken2 --db "$dirdb" --threads $THREADS --use-names --confidence 0.1 --report "$dirout/${name}_kraken2.tsv" --paired "$dir/${name}_nonrna_R1.fastq.gz" "$dir/${name}_nonrna_R2.fastq.gz" --output "$dirout/${name}_kraken2.txt"

        #Reestimação Bayesiana com Bracken
        bracken -d "$dirdb" -r 150 -i "$dirout/${name}_kraken2.tsv" -o "$dirout/${name}_bracken.tsv" -w "$dirout/${name}_reestimated.tsv" 

        echo "Classificação taxonômica concluída para $name"
    else
        echo "Nenhum arquivo encontrado em $dir"
    fi
done

echo "Todas as análises concluídas em $(date)"
echo "Até a próxima :)"