# Retraining Documentation

This guide explains how to prepare data and retrain either the **cytotoxicity classifier** or the **generative peptide model** on your own custom dataset.

## 1. Data Preparation

### 1.1 CSV File Format

All input files must be comma-separated values (`.csv`) with (at minimum) the following column:

| sequence                          | label |
|-----------------------------------|-------|
| FLPIGNLISALLRPLK                   | 1     |
| GLLRKGGEKIGEKLKEILG               | 0     |
| AVLVDKQCPD                          | 0     |
| ...                               | ...   |

**Required columns:**

- `sequence` — the amino acid sequence of the peptide (string, required for both models)
- `label`    — binary cytotoxicity label: `0` = non-cytotoxic, `1` = cytotoxic

**Important notes:**

- For **generative model** retraining  
  → the `label` column is **not required** and can be completely omitted
- For **cytotoxicity classifier** retraining  
  → the `label` column is **mandatory** and must contain only `0` or `1`

### 1.2 Required Folder Structure

Prepare one folder containing **exactly** these three files (case-sensitive names):

my_peptide_data/
├── train_set.csv
├── validation_set.csv
└── test_set.csv

## 2. Retraining the Cytotoxicity Classifier

Command:

```bash
python retrain_clf.py --training_folder /path/to/my_peptide_data
```

**Output directory:**

```
outputs/CTT_RETRAINING/
```

## 3. Retraining the Generative Model

Command:

```bash
python retrain_gen.py --training_folder /path/to/my_peptide_data
```

**Output directory:**

```
outputs/GEN_RETRAINING/
```

## 4. Using retrained models

To use new models, provide path via following flags:

   ```bash
   python run_pipeline.py --all_previous_flags --model_ctt path/to/classifier --model_gen path/to/generative_model
   ```