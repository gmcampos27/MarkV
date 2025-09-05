#!/bin/bash
set -euxo pipefail

# ========= CONFIG =========
BASE="/storage/vital/volume2/svetoslav_slavov/metaviromaHumano/Vic_nano/"
HPgV_REF="/storage/vital/volume2/svetoslav_slavov/metaviromaHumano/genomes/GBvC/ref.mmi"               # referência
CMV_REF="/storage/vital/volume2/svetoslav_slavov/metaviromaHumano/genomes/CMV/ref.mmi"
HIV_REF="/storage/vital/volume2/svetoslav_slavov/metaviromaHumano/genomes/hiv/ref.mni"
HAdVC_REF="/storage/vital/volume2/svetoslav_slavov/metaviromaHumano/genomes/HAdVC/ref.mmi"
THREADS=24
DATABASE="/storage/zuleika/volume3/database/kraken2.2/"               # banco de dados
HOST_REF="/storage/vital/volume2/svetoslav_slavov/metaviromaHumano/genomes/Homo_sapiens/GRCh38_latest_genomic.fna"
TAXID_HPgV=54290
TAXID_CMV=12345
TAXID_HIV=67890
TAXID_HAdVC=129951
# ==========================

for PROJECT in ${BASE}/*; do
    echo "=== Projeto: $(basename $PROJECT) ==="

    for BARCODE_DIR in ${PROJECT}/barcode*; do
        BARCODE=$(basename $BARCODE_DIR)
        echo "=== Processando ${BARCODE} em $(basename $PROJECT) ==="

        # 1) Merge dos fastq
        cat ${BARCODE_DIR}/*.fastq > ${BARCODE_DIR}/${BARCODE}_merged.fastq
        seqkit stats ${BARCODE_DIR}/${BARCODE}_merged.fastq > ${BARCODE_DIR}/${BARCODE}_stats.txt

        # 2) Filtragem de qualidade
        NanoFilt ${BARCODE_DIR}/${BARCODE}_merged.fastq -q 10 -l 101 \
            --logfile ${BARCODE_DIR}/${BARCODE}_NanoFilt.log \
            > ${BARCODE_DIR}/${BARCODE}_filtered.fastq

        # 3) Remover reads humanas
        minimap2 -t $THREADS -ax map-ont $HOST_REF ${BARCODE_DIR}/${BARCODE}_filtered.fastq \
            | samtools view -b -f 4 -o ${BARCODE_DIR}/${BARCODE}_unmapped.bam
        samtools fastq ${BARCODE_DIR}/${BARCODE}_unmapped.bam > ${BARCODE_DIR}/${BARCODE}_unmapped.fastq

        # 4) Classificação taxonômica
        kraken2 --db $DATABASE --threads $THREADS --use-names --confidence 0.1 \
            --report ${BARCODE_DIR}/${BARCODE}_kraken2.tsv \
            --output ${BARCODE_DIR}/${BARCODE}_kraken2.txt \
            ${BARCODE_DIR}/${BARCODE}_unmapped.fastq

        # 5) Análise específica dos vírus
        echo "=== [${BARCODE}] Checando Cytomegalovírus (CMV) (taxid $TAXID_CMV) ==="
        awk -v taxid=$TAXID_CMV '$0 ~ taxid {print $2}' ${BARCODE_DIR}/${BARCODE}_kraken2.txt > ${BARCODE_DIR}/${BARCODE}_cmv_ids.txt

        echo "=== [${BARCODE}] Checando HIV (taxid $TAXID_HIV) ==="
        awk -v taxid=$TAXID_HIV '$0 ~ taxid {print $2}' ${BARCODE_DIR}/${BARCODE}_kraken2.txt > ${BARCODE_DIR}/${BARCODE}_hiv_ids.txt

        echo "=== [${BARCODE}] Checando Human Adenovírus - C (HAdVC) (taxid $TAXID_HAdVC) ==="
        awk -v taxid=$TAXID_HAdVC '$0 ~ taxid {print $2}' ${BARCODE_DIR}/${BARCODE}_kraken2.txt > ${BARCODE_DIR}/${BARCODE}_hadv-c_ids.txt

        echo "=== [${BARCODE}] Checando Human Pegivirus (HPgV) (taxid $TAXID_HPgV) ==="
        awk -v taxid=$TAXID_HPgV '$0 ~ taxid {print $2}' ${BARCODE_DIR}/${BARCODE}_kraken2.txt > ${BARCODE_DIR}/${BARCODE}_hpgv_ids.txt

        if [[ -s ${BARCODE_DIR}/${BARCODE}_cmv_ids.txt || -s ${BARCODE_DIR}/${BARCODE}_hiv_ids.txt || -s ${BARCODE_DIR}/${BARCODE}_hadv-c_ids.txt || -s ${BARCODE_DIR}/${BARCODE}_hpgv_ids.txt ]]; then
            echo " IDs encontrados CMV:" $(wc -l < ${BARCODE_DIR}/${BARCODE}_cmv_ids.txt)
            echo " IDs encontrados HIV:" $(wc -l < ${BARCODE_DIR}/${BARCODE}_hiv_ids.txt)
            echo " IDs encontrados HAdV-C:" $(wc -l < ${BARCODE_DIR}/${BARCODE}_hadv-c_ids.txt)
            echo " IDs encontrados HPgV:" $(wc -l < ${BARCODE_DIR}/${BARCODE}_hpgv_ids.txt)

            # extrair reads
            seqtk subseq ${BARCODE_DIR}/${BARCODE}_unmapped.fastq ${BARCODE_DIR}/${BARCODE}_cmv_ids.txt > ${BARCODE_DIR}/${BARCODE}_cmv_reads.fastq
            seqtk subseq ${BARCODE_DIR}/${BARCODE}_unmapped.fastq ${BARCODE_DIR}/${BARCODE}_hiv_ids.txt > ${BARCODE_DIR}/${BARCODE}_hiv_reads.fastq
            seqtk subseq ${BARCODE_DIR}/${BARCODE}_unmapped.fastq ${BARCODE_DIR}/${BARCODE}_hadv-c_ids.txt > ${BARCODE_DIR}/${BARCODE}_hadv-c_reads.fastq
            seqtk subseq ${BARCODE_DIR}/${BARCODE}_unmapped.fastq ${BARCODE_DIR}/${BARCODE}_hpgv_ids.txt > ${BARCODE_DIR}/${BARCODE}_hpgv_reads.fastq

            # estatísticas básicas  
            seqkit stats ${BARCODE_DIR}/${BARCODE}_cmv_reads.fastq > ${BARCODE_DIR}/${BARCODE}_cmv_stats.txt
            seqkit stats ${BARCODE_DIR}/${BARCODE}_hiv_reads.fastq > ${BARCODE_DIR}/${BARCODE}_hiv_stats.txt
            seqkit stats ${BARCODE_DIR}/${BARCODE}_hadv-c_reads.fastq > ${BARCODE_DIR}/${BARCODE}_hadv-c_stats.txt
            seqkit stats ${BARCODE_DIR}/${BARCODE}_hpgv_reads.fastq > ${BARCODE_DIR}/${BARCODE}_hpgv_stats.txt

            # alinhamento contra ref do vírus
            #CMV, HIV, HAdV-C, HPgV
            minimap2 -ax map-ont -t $THREADS $CMV_REF ${BARCODE_DIR}/${BARCODE}_cmv_reads.fastq \
                | samtools sort -o ${BARCODE_DIR}/${BARCODE}_cmv_vs_ref.bam
            samtools index ${BARCODE_DIR}/${BARCODE}_cmv_vs_ref.bam
            samtools flagstat ${BARCODE_DIR}/${BARCODE}_cmv_vs_ref.bam > ${BARCODE_DIR}/${BARCODE}_cmv_vs_ref.flagstat.txt

            minimap2 -ax map-ont -t $THREADS $HIV_REF ${BARCODE_DIR}/${BARCODE}_hiv_reads.fastq \
                | samtools sort -o ${BARCODE_DIR}/${BARCODE}_hiv_vs_ref.bam
            samtools index ${BARCODE_DIR}/${BARCODE}_hiv_vs_ref.bam
            samtools flagstat ${BARCODE_DIR}/${BARCODE}_hiv_vs_ref.bam > ${BARCODE_DIR}/${BARCODE}_hiv_vs_ref.flagstat.txt

            minimap2 -ax map-ont -t $THREADS $HAdV-C_REF ${BARCODE_DIR}/${BARCODE}_hadv-c_reads.fastq \
                | samtools sort -o ${BARCODE_DIR}/${BARCODE}_hadv-c_vs_ref.bam
            samtools index ${BARCODE_DIR}/${BARCODE}_hadv-c_vs_ref.bam
            samtools flagstat ${BARCODE_DIR}/${BARCODE}_hadv-c_vs_ref.bam > ${BARCODE_DIR}/${BARCODE}_hadv-c_vs_ref.flagstat.txt

            minimap2 -ax map-ont -t $THREADS $HPgV_REF ${BARCODE_DIR}/${BARCODE}_hpgv_reads.fastq \
                | samtools sort -o ${BARCODE_DIR}/${BARCODE}_hpgv_vs_ref.bam
            samtools index ${BARCODE_DIR}/${BARCODE}_hpgv_vs_ref.bam
            samtools flagstat ${BARCODE_DIR}/${BARCODE}_hpgv_vs_ref.bam > ${BARCODE_DIR}/${BARCODE}_hpgv_vs_ref.flagstat.txt

        else
            echo "Nenhuma read atribuída aos taxids"
        fi

        echo "=== ${BARCODE} finalizado ==="

    done

    echo "=== Projeto $(basename $PROJECT) finalizado ==="

done

echo "Let us be devoured!"