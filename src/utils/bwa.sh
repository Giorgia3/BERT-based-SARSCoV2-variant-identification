# !/usr/bin/env sh
pwd
/home/giorgia/miniconda3/bin/bwa index datasets/reference.fasta
/home/giorgia/miniconda3/bin/bwa mem datasets/reference.fasta experiment/preprocessed_data/reformatted_seqs.fasta > experiment/preprocessed_data/aligned.sam
exit
