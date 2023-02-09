import math
from src.utils import general_config
import yaml
from pathlib import Path

with open(Path(".") / "config" / "bio_config.yaml") as bio_config_fp:
    bio_config_dict = yaml.safe_load(bio_config_fp)

spike_gene_start = bio_config_dict['spike_gene_start'] - 1
spike_gene_end = bio_config_dict['spike_gene_end'] - 1

domain_coordinates_1based = bio_config_dict['domain_coordinates_1based']

n_tokens_per_seq = math.ceil((((spike_gene_end-spike_gene_start) - general_config.K) / general_config.STRIDE) + 1)


def init():
    pass
