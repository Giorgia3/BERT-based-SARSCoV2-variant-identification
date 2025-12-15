# The geometry of BERT

This repository contains the main code and data associated with the article **"The geometry of BERT"** by Matteo Bonino, Giorgia Ghione, and Giansalvo Cirrincione. The article is available online at: https://arxiv.org/abs/2502.12033


## Abstract

Transformer neural networks, particularly Bidirectional Encoder Representations from Transformers (BERT), have shown remarkable performance across various tasks such as classification, text summarization, and question answering. However, their internal mechanisms remain mathematically obscure, highlighting the need for greater explainability and interpretability. In this direction, this paper investigates the internal mechanisms of BERT proposing a novel perspective on the attention mechanism of BERT from a theoretical perspective. The analysis encompasses both local and global network behavior. At the local level, the concept of directionality of subspace selection as well as a comprehensive study of the patterns emerging from the self-attention matrix are presented. Additionally, this work explores the semantic content of the information stream through data distribution analysis and global statistical measures including the novel concept of cone index. A case study on the classification of SARS-CoV-2 variants using RNA which resulted in a very high accuracy has been selected in order to observe these concepts in an application. The insights gained from this analysis contribute to a deeper understanding of BERT's classification process, offering potential avenues for future architectural improvements in Transformer models and further analysis in the training process. 


## Repository Contents
    .
    ├── config/                     # Configuration files
    ├── datasets/                   # Datasets and processed data
    ├── datasets_accession_numbers/ # GISAID data availability information
    ├── experiment/                 # Result folder
    ├── models/                     # Model files
    ├── src/                        # Source code
    ├── requirements.txt            # Python dependencies
    ├── LICENSE
    └── README.md

## Usage

Data preprocessing (requires BWA installed in `/root/miniconda3/bin/bwa`, see `src/utils/bwa.sh` script):
```bash
python src/main/preprocess_main.py --datasetsdir [Dataset directory] --bwadir [BWA directory]
```

The fine-tuned model can be downloaded here: [Link to the model](https://drive.google.com/file/d/1E69Jp-7TPuw1S7EWcOu6pw0NApLsFz9M/view?usp=sharing)

Inference and analyses (modify `TASK_TYPE` in `general_config.yaml` to select different analysis types):
```bash
python src/main/test_main.py --datasetsdir [Dataset directory] 
```

Analysis of attention patterns:
```bash
python src/main/attention_analysis_main.py --datasetsdir [Dataset directory] 
```

## Citation

If you reference or use this work, please cite:

    @misc{bonino2025geometrybert,
          title={The geometry of BERT}, 
          author={Matteo Bonino and Giorgia Ghione and Giansalvo Cirrincione},
          year={2025},
          eprint={2502.12033},
          archivePrefix={arXiv},
          primaryClass={cs.LG},
          url={https://arxiv.org/abs/2502.12033}, 
    }


## License
BERT-based-SARSCoV2-variant-identification is distributed under a GPL-3.0 license.
