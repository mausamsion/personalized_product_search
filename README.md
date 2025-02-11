## Paper Implementation

PyTorch implementation of the paper [A Transformer-based Embedding Model for Personalized Product Search](https://arxiv.org/pdf/2005.08936)

### Repo structure
```
.
├── LICENSE
├── README.md
├── data
│   ├── cell-phones_test.tsv
│   ├── cell-phones_train.tsv
│   ├── item_tokens.parquet
│   ├── item_vocab.json
│   └── token_vocab.json
├── nbs
│   ├── interaction_data.ipynb
│   ├── item_eda.ipynb
│   └── query_eda.ipynb
└── src
    ├── config.yaml
    ├── item_language_model.py
    └── pps_model.py
```

- Complete pipeline code of the paper is in `src/pps_model.py`.
- `src/item_language_model.py` only trains a item language model separately.
- `nbs` contains the work with dataset analysis.
- `data` has some intermediate files saved from the dataset analysis.

