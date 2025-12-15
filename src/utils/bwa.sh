# !/usr/bin/env sh
pwd
/root/miniconda3/bin/bwa index datasets/reference.fasta
/root/miniconda3/bin/bwa mem datasets/reference.fasta experiment/preprocessed_data/reformatted_seqs.fasta > experiment/preprocessed_data/aligned.sam
exit
