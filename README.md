# MS Thesis: An interpretable BERT-based architecture for SARS-CoV-2 variant identification

This repository contains the main code and data associated with the MS thesis:

**"An interpretable BERT-based architecture for SARS-CoV-2 variant identification"** \
Giorgia Ghione\
MS Degree in Computer Engineering\
Politecnico di Torino, 2022\
Supervisors: Santa Di Cataldo, Marta Lovino, Giansalvo Cirrincione, Elisa Ficarra

The full thesis is publicly available at: https://webthesis.biblio.polito.it/23527/


## Abstract

The Covid-19 pandemic has posed many challenges in the medical diagnostics field. One of these has been the need for constant detection and monitoring of the SARS-CoV-2 circulating variants. The most common approach to reliably identify a SARS-CoV-2 variant is exploiting genomics. Such an approach has been enabled by the constant collection of genetic sequences of the virus globally. However, variant identification methods are usually resource-intensive. Thus, small medical laboratories can have issues due to limited diagnostic capacity. This thesis presents a deep learning method to successfully identify variants without requiring high computational resources and long delays. The contribution of this thesis is twofold: 1) the development of a Bidirectional Encoder Representations from Transformers (BERT) fine-tuning architecture for SARS-CoV-2 variant identification; 2) the mathematical and biological interpretation of the model by leveraging its self-attention mechanism. The developed method allows the analysis of the spike gene of SARS-CoV-2 genome samples to determine their variant quickly. The chosen neural network BERT is a Transformer-based model initially proposed for processing natural language sequences. However, it has been successfully applied to several other contexts, such as DNA/RNA sequence analysis. Therefore, BERT was fine-tuned to adapt to the genomic sequence domain, reaching an F1 score equal to 98.59% on the inference dataset: it proved effective in recognizing variants circulating to date. Since BERT relies on the self-attention mechanism, the interpretability of the model was investigated by analyzing its self-attention matrices and hidden weights. The resulting mathematical interpretation allowed the understanding of the biological meaning of the attention patterns produced by the network. Indeed, BERT extracts relevant biological information on variants by focusing on specific parts of the SARS-CoV-2 spike gene. In particular, it was examined how attention spreads across the domains of the spike protein, and it was found that attention is often localized on the site of defining mutations of variants. Therefore, the developed architecture allows gaining insights into the distinctive characteristics of SARS-CoV-2 genetic sequences and into the behaviour of BERT neural network.

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

```bash
python src/main/preprocess_main.py --datasetsdir [Dataset directory] --bwadir [BWA directory]
python src/main/test_main.py --datasetsdir [Dataset directory] 
python src/main/attention_analysis_main.py --datasetsdir [Dataset directory] 
```

## Citation
  
    @mastersthesis{ghione2022interpretable,
      title={An interpretable BERT-based architecture for SARS-CoV-2 variant identification},
      author={Ghione, Giorgia},
      year={2022},
      school={Politecnico di Torino}
    }


## License
BERT-based-SARSCoV2-variant-identification is distributed under a GPL-3.0 license.
