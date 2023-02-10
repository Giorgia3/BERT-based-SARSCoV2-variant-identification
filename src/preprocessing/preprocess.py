import csv
import os
from pathlib import Path
from tqdm import tqdm
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import pandas as pd
import matplotlib.pyplot as plt
import pysam
from src.utils import general_utils


def read_fasta(fp):
    """
    Read fasta file, yield id and sequence.
    """
    id, seq = None, []
    for line in fp:
        line = line.rstrip()
        if line.startswith(">"):
            if id: yield id, ''.join(seq)
            id, seq = line, []
        else:
            seq.append(line)
    if id: yield id, ''.join(seq)


def split_fixed_len_chunks(seq, chunk_len, chunk_stride):
    """
    Split sequence in fixed-length chunks with stride. Pad shorter chunks with 'N'.
    """
    chunks = []
    for i in range(0, len(seq), chunk_stride):
        window = seq[i:i + chunk_len]
        window += 'N' * (chunk_len - len(window))
        chunks.append(window)
    return chunks


def reformat_fasta(config, input_file, output_file, duplicates_file, label, log_fp_trainval):
    """
    Read fasta file, possibily split each sequence into chunks of length CHUNK_LEN and
    reformat: '[label], [id], [base1], [base2], ..., [baseN]'
    """
    ids_dict = {}

    index = []
    seq_len_info = {}
    seq_len_info['max'] = 0
    seq_len_info['min'] = None
    seq_len_info['sum'] = 0
    seq_len_info['avg'] = 0
    seq_len_info['n_seq'] = 0
    seq_len_info['n_dup'] = 0

    count = 0

    with open(input_file) as in_fp, open(output_file, 'a') as out_file, open(duplicates_file, 'a') as dup_file:

        for seq_id, seq in tqdm(read_fasta(in_fp)):

            # remove duplicates
            if seq_id in ids_dict:
                s = str(label) + ',' + str(seq_id) + ',' + str(0) + ',' + ''.join(seq) + '\n'
                dup_file.write(s)
                seq_len_info['n_dup'] += 1
                continue

            # convert patient id from string to int
            ids_dict[seq_id] = [len(ids_dict) + 1, label]
            id = ids_dict[seq_id][0]

            if config.general_config.SPLIT_DATA_IN_CHUNKS or config.general_config.ALIGNMENT:
                seq_len_info['n_seq'] += 1

                # split sequence in chunks
                chunks = split_fixed_len_chunks(seq, config.general_config.CHUNK_LEN, config.general_config.CHUNK_STRIDE)

                # reformat chunk info and write on file
                # (NB: each chunk is identified by its sequence id and its position in the sequence)
                if config.general_config.ALIGNMENT:
                    seq_records = []
                    for pos, c in enumerate(chunks):
                        # FASTA format
                        header = str(label) + ',' + str(id) + ',' + str(pos)
                        s = Seq(''.join(c))
                        s_record = SeqRecord(s, id=header, description="")
                        seq_records.append(s_record)
                    SeqIO.write(seq_records, out_file, "fasta")
                else:
                    for pos, c in enumerate(chunks):
                        # CSV format
                        s = str(label) + ',' + str(id) + ',' + str(pos) + ',' + ''.join(c) + '\n'
                        out_file.write(s)

                # update length statistics
                index.append([id, label])
                seq_len_info['sum'] += len(chunks)
                if len(chunks) > seq_len_info['max']:
                    seq_len_info['max'] = len(chunks)
                if seq_len_info['min'] == None or len(chunks) < seq_len_info['min']:
                    seq_len_info['min'] = len(chunks)

            else:
                # reformat sequence info and write on file
                s = str(label) + ',' + str(id) + ',' + str(0) + ',' + ''.join(seq) + '\n'
                out_file.write(s)
                index.append([id, label])

                # update length statistics
                seq_len_info['sum'] += len(seq)
                seq_len_info['n_seq'] += 1
                if len(seq) > seq_len_info['max']:
                    seq_len_info['max'] = len(seq)
                if seq_len_info['min'] == None or len(seq) < seq_len_info['min']:
                    seq_len_info['min'] = len(seq)

            # stop earlier to consider only MAX_N_SAMPLES_PER_CLASS samples per class
            count += 1
            if config.general_config.REDUCE_N_INPUT_SAMPLES and count >= config.general_config.MAX_N_SAMPLES_PER_CLASS:
                break

        # print length statistics
        seq_len_info['avg'] = seq_len_info['sum'] / seq_len_info['n_seq']
        print(f"\tN. sequences: {seq_len_info['n_seq']}")
        print(f"\tN. duplicated sequences: {seq_len_info['n_dup']}")
        print(f"\tTot. n. chunks: {seq_len_info['sum']}")
        print(
            f"\tMin. {'n. chunks per sequence' if config.general_config.SPLIT_DATA_IN_CHUNKS else 'len'}: {seq_len_info['min']}")
        print(
            f"\tMax. {'n. chunks per sequence' if config.general_config.SPLIT_DATA_IN_CHUNKS else 'len'}: {seq_len_info['max']}")
        print(
            f"\tAvg. {'n. chunks per sequence' if config.general_config.SPLIT_DATA_IN_CHUNKS else 'len'}: {seq_len_info['avg']}")
        print()
        log_fp_trainval.write(f"\tN. sequences: {seq_len_info['n_seq']}\n")
        log_fp_trainval.write(f"\tN. duplicated sequences: {seq_len_info['n_dup']}\n")
        log_fp_trainval.write(f"\tTot. n. chunks: {seq_len_info['sum']}\n")
        log_fp_trainval.write(
            f"\tMin. {'n. chunks per sequence' if config.general_config.SPLIT_DATA_IN_CHUNKS else 'len'}: {seq_len_info['min']}\n")
        log_fp_trainval.write(
            f"\tMax. {'n. chunks per sequence' if config.general_config.SPLIT_DATA_IN_CHUNKS else 'len'}: {seq_len_info['max']}\n")
        log_fp_trainval.write(
            f"\tAvg. {'n. chunks per sequence' if config.general_config.SPLIT_DATA_IN_CHUNKS else 'len'}: {seq_len_info['avg']}\n\n")

    with open(config.paths_config.ids_dict_file, 'w') as ids_dict_fp:
        ids_dict_fp.write(f"seq_id,id,label\n")
        for seq_id, id_label_list in ids_dict.items():
            ids_dict_fp.write(f"{seq_id},{id_label_list[0]},{id_label_list[1]}\n")

    return index, seq_len_info


def reformat_input_data(config):
    print("Reformatting input data")
    # check if all input paths exist
    for _, input_file in config.paths_config.variant_files.items():
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"File {input_file} not found")

    # if reformatted data already available continue, else reformat
    if os.path.exists(config.paths_config.reformatted_seqs_file):
        print(
            f"Data already reformatted in {config.paths_config.reformatted_seqs_file} file. See {config.paths_config.log_file} for data info.")
    else:
        with open(config.paths_config.seqs_index_file_tmp, 'w') as seqs_index_fp, open(config.paths_config.log_file, 'w') as log_fp:
            tot_n_seq = 0
            tot_sum_len = 0
            tot_max_len_list = []
            tot_min_len_list = []

            log_fp.write(f"Sequences info:\n")
            log_fp.write(f"===============\n")

            # read each variant input file and reformat, possibly splitting sequences into chunks
            for variant_name, input_file in config.paths_config.variant_files.items():
                print(
                    f"Variant name: {variant_name} | Label: {config.general_config.CLASS_LABELS[variant_name]} | File: {input_file}")
                log_fp.write(
                    f"Variant name: {variant_name} | Label: {config.general_config.CLASS_LABELS[variant_name]} | File: {input_file}\n")

                variant_seqs_index, variant_seq_len_info = reformat_fasta(config,
                                                                          input_file,
                                                                          config.paths_config.reformatted_seqs_file,
                                                                          config.paths_config.duplicated_seqs_file,
                                                                          config.general_config.CLASS_LABELS[variant_name],
                                                                          log_fp)

                seqs_index_csvwriter = csv.writer(seqs_index_fp)
                seqs_index_csvwriter.writerows(variant_seqs_index)

                tot_max_len_list.append(variant_seq_len_info['max'])
                tot_min_len_list.append(variant_seq_len_info['min'])
                tot_sum_len += variant_seq_len_info['sum']
                tot_n_seq += variant_seq_len_info['n_seq']

            tot_max_len = max(tot_max_len_list)
            tot_min_len = min(tot_min_len_list)

            print(f'Tot info:')
            print(f'\tN. seqences: {tot_n_seq}')
            print(f'\tTot n. chunks: {tot_sum_len}')
            print(f"\tMin {'n. chunks' if config.general_config.SPLIT_DATA_IN_CHUNKS else 'len'}: {tot_min_len}")
            print(f"\tMax {'n. chunks' if config.general_config.SPLIT_DATA_IN_CHUNKS else 'len'}: {tot_max_len}")
            print(f"\tAvg {'n. chunks' if config.general_config.SPLIT_DATA_IN_CHUNKS else 'len'}: {tot_sum_len / tot_n_seq}")
            log_fp.write(f'Tot info:\n')
            log_fp.write(f'\tN. seqences: {tot_n_seq}\n')
            log_fp.write(
                f"\tMin {'n. chunks' if config.general_config.SPLIT_DATA_IN_CHUNKS else 'len'}: {min(tot_min_len_list)}\n")
            log_fp.write(
                f"\tMax {'n. chunks' if config.general_config.SPLIT_DATA_IN_CHUNKS else 'len'}: {max(tot_max_len_list)}\n")
            log_fp.write(
                f"\tAvg {'n. chunks' if config.general_config.SPLIT_DATA_IN_CHUNKS else 'len'}: {tot_sum_len / tot_n_seq}\n")
            log_fp.write('--------------------------------------------------------------\n')
    print("Done")

def __fetch_aligned_spike_seqs(config, aligned_seqs_file, class_num):
    if not os.path.exists(aligned_seqs_file):
        print(f"Error: file {aligned_seqs_file} not found.")
    print(f'Processing aligned sequences of class {class_num}.')

    aligned_seqs_dict = {}
    samfile = pysam.AlignmentFile(aligned_seqs_file, "r")
    i = 0
    n_mapped = 0
    n_reverse = 0
    n_supplementary = 0
    n_secondary = 0
    n_unmapped = 0
    n_tot_flagged = 0

    # parse sam file
    for aligned_segment in tqdm(samfile.fetch()):

        # check if read correctly aligned or reverse
        if aligned_segment.flag == 0 or aligned_segment.is_reverse:

            qname = aligned_segment.query_name.split(',')
            label = qname[0]
            id = qname[1]

            # select only sequences of current class
            if label == str(class_num):

                aligned_chunk = aligned_segment.query_alignment_sequence
                aligned_pos_start = aligned_segment.reference_start  # 0-based leftmost coordinate
                aligned_pos_end = aligned_segment.reference_end - 1  # reference_end points to one past the last aligned residue
                aligned_reference_positions = aligned_segment.get_reference_positions(
                    full_length=True)  # a list of reference positions that this read aligns to.
                # If full_length is set, None values will be included for any soft-clipped or unaligned positions within the read.
                # The returned list will thus be of the same length as the read
                cigar = aligned_segment.cigarstring
                aligned_reference_positions_spike = []

                n_mapped += 1
                # check if read mapped over a portion of the spike gene
                if (aligned_pos_start < config.bio_config.spike_gene_start and aligned_pos_end > config.bio_config.spike_gene_start) \
                        or (aligned_pos_start > config.bio_config.spike_gene_start and aligned_pos_end < config.bio_config.spike_gene_end) \
                        or (aligned_pos_start < config.bio_config.spike_gene_end and aligned_pos_end > config.bio_config.spike_gene_end):

                    # print(f'{qname} {aligned_pos_start} {aligned_pos_end}')
                    # i +=1

                    # save read info in dict, key is id
                    if id not in aligned_seqs_dict:
                        aligned_seqs_dict[id] = {'label': label,
                                                 'pos_start_list': [],
                                                 'pos_end_list': [],
                                                 'reference_positions_list': [],
                                                 'aligned_chunk_spike_list': [],
                                                 'cigar_list': []}
                    aligned_seqs_dict[id]['cigar_list'].append(cigar)

                    # reverse complement if chunk has been aligned as reversed
                    if aligned_segment.is_reverse:
                        n_reverse += 1
                        aligned_chunk = str(Seq(aligned_chunk).reverse_complement())

                    # truncate chunks with bases outside of the spike gene
                    # 1st case: first bases of read mapped outside of spike region
                    if aligned_pos_start < config.bio_config.spike_gene_start < aligned_pos_end:
                        # truncate aligned chunk
                        aligned_chunk_spike = aligned_chunk[config.bio_config.spike_gene_start - aligned_pos_start:]
                        # modify start position
                        aligned_pos_start = config.bio_config.spike_gene_start
                        # truncate reference positions of aligment
                        for idx, pos in enumerate(aligned_reference_positions):
                            if pos != None and pos >= config.bio_config.spike_gene_start:
                                aligned_reference_positions_spike = aligned_reference_positions[idx:]
                                break

                    # 2nd case: last bases of read mapped outside of spike region
                    elif aligned_pos_start < config.bio_config.spike_gene_end < aligned_pos_end:
                        # truncate aligned chunk
                        aligned_chunk_spike = aligned_chunk[:config.bio_config.spike_gene_end - aligned_pos_start + 1]
                        # modify end position
                        aligned_pos_end = config.bio_config.spike_gene_end
                        # truncate reference positions of aligment
                        for idx, pos in enumerate(aligned_reference_positions):
                            if pos != None and pos > config.bio_config.spike_gene_end:
                                aligned_reference_positions_spike = aligned_reference_positions[:idx]
                                break

                    # 3rd case: read completely mapped inside spike region
                    else:
                        aligned_chunk_spike = aligned_chunk
                        aligned_reference_positions_spike = aligned_reference_positions

                    aligned_seqs_dict[id]['aligned_chunk_spike_list'].append(aligned_chunk_spike)
                    aligned_seqs_dict[id]['reference_positions_list'].append(aligned_reference_positions_spike)
                    relative_aligned_pos_start = aligned_pos_start - config.bio_config.spike_gene_start
                    aligned_seqs_dict[id]['pos_start_list'].append(relative_aligned_pos_start)
                    relative_aligned_pos_end = aligned_pos_end - config.bio_config.spike_gene_start
                    aligned_seqs_dict[id]['pos_end_list'].append(relative_aligned_pos_end)

                    # if (relative_aligned_pos_end-relative_aligned_pos_start+1) != len(aligned_chunk_spike):
                    #     print(f"Attention: {relative_aligned_pos_end}-{relative_aligned_pos_start}+1={relative_aligned_pos_end-relative_aligned_pos_start} != {len(aligned_chunk_spike)}")
                    #     print(aligned_segment.get_reference_positions(full_length=True))
                    # else:
                    #     print('OK')

        # check if this is a supplementary alignment
        if aligned_segment.is_supplementary:
            n_supplementary += 1
        # check if not primary alignment
        if aligned_segment.is_secondary:
            n_secondary += 1
        # check if read itself is unmapped
        if aligned_segment.is_unmapped:
            n_unmapped += 1
        if aligned_segment.flag != 0:
            n_tot_flagged += 1

        # if i>10:
        #     break

    samfile.close()

    # write sequences index on file
    with open(config.paths_config.seqs_index_file, 'a') as seqs_index_fp:
        seqs_index = [[int(id), aligned_seqs_dict[id]['label']] for id in aligned_seqs_dict]
        seqs_index_csvwriter = csv.writer(seqs_index_fp)
        seqs_index_csvwriter.writerows(seqs_index)

    print(f'Skipped chunks and actions for class {class_num}:')
    print(f'\tn_reverse={n_reverse} (reverse complemented)')
    print(f'\tn_supplementary={n_supplementary} (skipped)')
    print(f'\tn_secondary={n_secondary} (skipped)')
    print(f'\tn_unmapped={n_unmapped} (skipped)')
    print(f'\tn_tot_flagged={n_tot_flagged}')

    with open(config.paths_config.log_file, 'a') as log_fp:
        log_fp.write(f"Alignment of the Spike gene\n")
        log_fp.write(f"============================\n")
        log_fp.write(f'Skipped chunks and actions for class {class_num}:\n')
        log_fp.write(f'\tn_reverse={n_reverse} (reverse complemented)\n')
        log_fp.write(f'\tn_supplementary={n_supplementary} (skipped)\n')
        log_fp.write(f'\tn_secondary={n_secondary} (skipped)\n')
        log_fp.write(f'\tn_unmapped={n_unmapped} (skipped)\n')
        log_fp.write(f'\tn_tot_flagged={n_tot_flagged}\n')

    return aligned_seqs_dict  # , seqs_index


def __concat_spike_seqs_on_file(config, aligned_seqs_dict, cigars_file, reference_seq_file):
    # get reference sequence
    with open(reference_seq_file, 'r') as reference_seq_fp:
        for _, reference_seq in read_fasta(reference_seq_fp):
            break
        reference_seq_spike = reference_seq[config.bio_config.spike_gene_start:config.bio_config.spike_gene_end + 1]

    with open(config.paths_config.spike_seqs_file, 'a') as spike_seqs_fp, open(cigars_file, 'a') as cigars_fp:

        # concatenated_alignments = reference_seq.copy()

        # get data of each patient
        for id in tqdm(aligned_seqs_dict):
            pos_start_list = aligned_seqs_dict[id]['pos_start_list']
            pos_end_list = aligned_seqs_dict[id]['pos_end_list']
            reference_positions_list = aligned_seqs_dict[id]['reference_positions_list']
            aligned_chunk_spike_list = aligned_seqs_dict[id]['aligned_chunk_spike_list']
            cigar_list = aligned_seqs_dict[id]['cigar_list']

            # sort list of alignments of that id based on the start position
            pos_start_list_sorted, pos_end_list_sorted, aligned_chunk_spike_list_sorted, cigar_list_sorted = \
                (list(l) for l in zip(*sorted(zip(pos_start_list, pos_end_list, aligned_chunk_spike_list, cigar_list),
                                              key=lambda x: x[0])))
            pos_start_list_sorted = pos_start_list
            pos_end_list_sorted = pos_end_list
            aligned_chunk_spike_list_sorted = aligned_chunk_spike_list

            concatenated_alignment = []
            prev_pos_end = None
            is_start = True
            for pos_start, pos_end, reference_positions, aligned_chunk_spike in zip(pos_start_list_sorted,
                                                                                    pos_end_list_sorted,
                                                                                    reference_positions_list,
                                                                                    aligned_chunk_spike_list_sorted):
                read_seq = []

                # check if first bases of reference are unmapped
                if is_start and pos_start > 0:
                    # copy reference bases
                    concatenated_alignment.extend(reference_seq_spike[:pos_start])

                # check if there is a gap of unmapped bases between previous read and current read
                if not is_start and pos_start - prev_pos_end > 1:
                    concatenated_alignment.extend(reference_seq_spike[prev_pos_end:pos_start])

                is_start = False

                concatenated_alignment.extend(aligned_chunk_spike)

                prev_pos_end = pos_end

            # check if the last bases of reference are unmapped
            if config.bio_config.spike_gene_end - prev_pos_end > 0:
                concatenated_alignment.extend(reference_seq_spike[prev_pos_end:config.bio_config.spike_gene_end + 1])

            s = str(aligned_seqs_dict[id]['label']) + ',' + str(id) + ',' + str(0) + ',' + ''.join(
                concatenated_alignment) + '\n'
            spike_seqs_fp.write(s)

            # TODO write cigars file


def __run_bwa(config):
    print("Running BWA")
    bwa_exe_path = str(Path(config.paths_config.bwa_bin_dir) / 'bwa')
    bwa_script_path = Path(config.paths_config.main_dir) / 'src' / 'utils' / 'bwa.sh'

    #generate script
    with open(bwa_script_path, 'w') as bwa_script_fp:
        bwa_script_fp.write(f"# !/usr/bin/env sh\n")
        bwa_script_fp.write(f"pwd\n")
        bwa_script_fp.write(f"{bwa_exe_path} index {config.paths_config.reference_seq_file}\n")
        bwa_script_fp.write(f"{bwa_exe_path} mem {config.paths_config.reference_seq_file} {config.paths_config.reformatted_seqs_file} > {config.paths_config.aligned_seqs_file}\n")
        bwa_script_fp.write(f"exit\n")
    os.system(f'wsl.exe -e sh -c {bwa_script_path}')

    # cmd_generate_idx = f'wsl.exe ~ -e sh -c "{bwa_exe_path} index {config.paths_config.reference_seq_file}"'
    # cmd_align = f'wsl.exe ~ -e sh -c  "{bwa_exe_path} mem {config.paths_config.reference_seq_file} {config.paths_config.reformatted_seqs_file} > {config.paths_config.aligned_seqs_file}"'
    # os.system(cmd_generate_idx)
    # os.system(cmd_align)


def align_spike_sequences(config):
    if os.path.exists(config.paths_config.aligned_seqs_file):
        print(f"BWA already performed. SAM file available: {config.paths_config.aligned_seqs_file}")
    else:
        __run_bwa(config)

    if config.general_config.SPIKE_REGION_ANALYSIS:
        print("Extracting spike region")
        if os.path.exists(config.paths_config.spike_seqs_file):
            print(f"Aligned spike gene sequences already available in {config.paths_config.spike_seqs_file} file.")
        else:
            __run_bwa(config)
            for class_num in range(len(config.general_config.CLASS_LABELS.keys())):
                aligned_seqs_dict = __fetch_aligned_spike_seqs(config, config.paths_config.aligned_seqs_file, class_num)
                __concat_spike_seqs_on_file(config, aligned_seqs_dict, config.paths_config.cigars_file, config.paths_config.reference_seq_file)
                del aligned_seqs_dict
    print("Done")


# def find_duplicated_seqs(filepath, seqs_index_dict, type_dataset, remove_dups=False):
def __find_duplicated_seqs(config, filepath, seqs_index_dict, remove_dups=False):
    # if type_dataset not in ['train', 'val', 'test']:
    #     raise Exception('Not valid type_dataset: select train, val or test')
    title = f'total duplicates'
    tot_dups = 0
    tot_seqs = 0
    duplicates_line_num = []
    with open(config.paths_config.log_file, 'a') as log_fp:
        log_fp.write(f"{title}\n")
        log_fp.write(f"==============================\n")

        with open(filepath, 'r') as fp:
            for class_lab, class_seq_nums in seqs_index_dict.items():
                class_seqs = []
                class_ids = []
                class_line_nums = []
                csv_reader = csv.reader(fp, delimiter=',')
                for n_line, line in enumerate(csv_reader):
                    label = line[0]
                    if label == str(class_lab):
                        seq = line[3]
                        id = line[1]
                        class_seqs.append(seq)
                        class_ids.append(id)
                        class_line_nums.append(n_line)
                        # class_lines.append(line)
                fp.seek(0)
                class_seqs_df = pd.DataFrame({'seqs': class_seqs,
                                              'ids': class_ids,
                                              'line_nums': class_line_nums})
                del class_seqs
                del class_ids
                del class_line_nums
                dup_seqs = class_seqs_df['seqs'].duplicated()
                dup_ids = class_seqs_df['ids'].duplicated()
                dup_mask = dup_seqs | dup_ids
                duplicates_line_num.extend(class_seqs_df[dup_mask == True]['line_nums'])
                tot_dups += dup_mask.sum()
                tot_seqs += len(dup_mask)

                log_dups = f'{general_utils.get_inverted_class_labels_dict(config)[class_lab]} | n. duplicated seqs = {dup_seqs.sum()}/{len(dup_seqs)} ({dup_seqs.sum() / len(dup_seqs) * 100:.2f}%) | n. duplicated ids = {dup_ids.sum()}/{len(dup_ids)} ({dup_ids.sum() / len(dup_ids) * 100:.2f}%)'
                log_fp.write(f"{log_dups}\n")
                print(log_dups)

            log_dups = f'Tot n. duplicates = {tot_dups}/{tot_seqs}\t({tot_dups / tot_seqs * 100:.2f}%)'
            log_fp.write(f"{log_dups}\n\n")
            print(f"{log_dups}\n")

        if remove_dups:
            if tot_dups != 0:
                print(f"Removing {title}...")
                dirname = os.path.dirname(filepath)
                filename = os.path.splitext(os.path.basename(filepath))[0]
                filepath_no_dups = Path(dirname) / f'{filename}_no_dups.csv'
                with open(filepath, 'r') as fp, open(filepath_no_dups, 'w') as fp_no_dups, open(config.paths_config.seqs_index_file,
                                                                                                'w') as seqs_index_fp:
                    fp_no_dups_csvwriter = csv.writer(fp_no_dups)
                    seqs_index_csvwriter = csv.writer(seqs_index_fp)
                    csv_reader = csv.reader(fp, delimiter=',')
                    for n_line, line in enumerate(csv_reader):
                        if n_line not in duplicates_line_num:
                            fp_no_dups_csvwriter.writerow(line)
                            index = [int(line[1]), int(line[0])]
                            seqs_index_csvwriter.writerow(index)
                os.rename(filepath, f'{filepath}_with_dups.csv')
                os.rename(filepath_no_dups, filepath)
                print('Done.\n')
                log_fp.write(f"Removed {title}\n\n")

                # #update data sizes
                # sizes_info[f'total_data_size_seqs'] = sum(1 for line in open(test_file))
                # with open(trainvaltest_sizes_file, 'w') as trainvaltest_sizes_fp:
                #     for key in sizes_info.keys():
                #         trainvaltest_sizes_fp.write(f"{key},{sizes_info[key]}\n")

            else:
                print(f"No {title} to remove.")


def read_seqs_index(seqs_index_file):
    seqs_index = []
    with open(seqs_index_file, 'r') as seqs_index_fp:
        seqs_index_csvreader = csv.reader(seqs_index_fp, delimiter=',')
        for line in tqdm(seqs_index_csvreader):
            seqs_index.append([int(line[0]), int(line[1])])
    return seqs_index


def check_duplicates(config):
    print("Searching for duplicates")
    seqs_index_dict = {}
    seqs_index = read_seqs_index(config.paths_config.seqs_index_file)

    for seq_n, seq_lab in seqs_index:
        if seq_lab not in seqs_index_dict:
            seqs_index_dict[seq_lab] = []
        seqs_index_dict[seq_lab].append(seq_n)

    __find_duplicated_seqs(config, config.paths_config.input_seqs_file, seqs_index_dict, remove_dups=True)

    # find_duplicated_seqs(train_file, seqs_index_dict, type_dataset="train")
    # find_duplicated_seqs(val_file, seqs_index_dict, type_dataset="val")
    # find_duplicated_seqs(test_file, seqs_index_dict, type_dataset="test", remove_dups=True)

    # print dups info
    df_dups = pd.read_csv(Path(config.paths_config.preprocessed_data_dir) / 'dups_info.txt', delimiter=',', header=0)
    df_dups['N. not duplicated'] = df_dups['tot'] - df_dups['dups']
    df_dups.rename(columns={'dups': 'N. duplicates', 'class': 'Class'}, inplace=True)
    df_dups_plot = df_dups.drop(columns=['percent', 'tot'])
    df_dups_plot.plot(x='Class', kind='bar', stacked=True, title='Stacked Bar Graph by dataframe',
                      ylabel='N. sequences')
    plt.show()
    print("Done")


def write_test_size_info(config):
    seqs_index = read_seqs_index(config.paths_config.seqs_index_file)
    with open(config.paths_config.trainvaltest_sizes_file, 'w') as trainvaltest_sizes_fp:
        trainvaltest_sizes_fp.write(f"test_data_size_seqs,{len(seqs_index)}\n")