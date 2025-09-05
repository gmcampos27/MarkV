# Define an array of virus names and their corresponding reference genomes
declare -a viruses=(
    "respirovirus3:../genomes/respirovirus/respirovirus3.fasta:../QC/Alagoas/Pool_4_R1_trimmed.fastq.gz:../QC/Alagoas/Pool_4_R2_trimmed.fastq.gz"
    "HumanHerpesVirus4:../genomes/EBV/EBV.fasta:../QC/Alagoas/Pool_2_R1_trimmed.fastq.gz:../QC/Alagoas/Pool_2_R2_trimmed.fastq.gz"
    "EVJ:../genomes/EnterovirusJ/EVJ.fasta:../QC/Alagoas/Pool_2_R1_trimmed.fastq.gz:../QC/Alagoas/Pool_2_R2_trimmed.fastq.gz"
    "RhinovirusC:../genomes/RhinovirusNAT/nat.fasta:../QC/Alagoas/Pool_5_R1_trimmed.fastq.gz:../QC/Alagoas/Pool_5_R2_trimmed.fastq.gz"
    "HumanHerpes6:../genomes/Herpes6/HHV6.fasta:../QC/Alagoas/Pool_6_R1_trimmed.fastq.gz:../QC/Alagoas/Pool_6_R2_trimmed.fastq.gz"
    "Coronavirus:../genomes/HumanCov/HCoVOC43.fasta:../QC/Alagoas/Pool_7_R1_trimmed.fastq.gz:../QC/Alagoas/Pool_7_R2_trimmed.fastq.gz"
    "Parvovirus:../genomes/HumanParvovirus/HPVB19.fasta:../QC/Alagoas/Pool_8_R1_trimmed.fastq.gz:../QC/Alagoas/Pool_8_R2_trimmed.fastq.gz"
    "HumanHerpes7:../genomes/Herpes7/HHV7.fasta:../QC/Alagoas/Pool_9_R1_trimmed.fastq.gz:../QC/Alagoas/Pool_9_R2_trimmed.fastq.gz"
    "TTV3:../genomes/TTV3/TTV3.fasta:../QC/Alagoas/Pool_10_R1_trimmed.fastq.gz:../QC/Alagoas/Pool_10_R2_trimmed.fastq.gz"
    #"respirovirus3:../genomes/respirovirus/respirovirus3.fasta:../QC/Alagoas/Pool_4_R1_trimmed.fastq.gz:../QC/Alagoas/Pool_4_R1_trimmed.fastq.gz"
    # Add more viruses as needed in the format "virus_name:viral_reference_genome.fasta:metagenomic_reads_virus.fastq"
)

# Iterate over each virus
for virus_info in "${viruses[@]}"; do
    # Split the virus_info into virus name, reference genome filename, R1 reads filename, and R2 reads filename
    IFS=':' read -r -a info_parts <<< "$virus_info"
    virus_name="${info_parts[0]}"
    reference_genome="${info_parts[1]}"
    reads_file_R1="${info_parts[2]}"
    reads_file_R2="${info_parts[3]}"

echo "Processing $virus_name..."

    # Step 1: Index the reference genome
    bwa index -a bwtsw "$reference_genome"

    # Step 2: Map reads to the reference genome
    bwa mem -t 8 -P "$reference_genome" "$reads_file_R1" "$reads_file_R2" | samtools view -b -F 4 > "mapping_${virus_name}.bam"

    # Step 3: Sort the BAM file
    samtools sort "mapping_${virus_name}.bam" -o "mapping_sorted_${virus_name}.bam"

    # Step 4: Index the sorted BAM file
    samtools index "mapping_sorted_${virus_name}.bam"

    # Step 5: Calculate depth of coverage
    samtools depth -a "mapping_sorted_${virus_name}.bam" > "coverage_depth_${virus_name}.txt"

    echo "Processing $virus_name complete."
done

echo "All viruses processed."
