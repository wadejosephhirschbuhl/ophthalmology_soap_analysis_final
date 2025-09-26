# Ophthalmology Note Generation and Robustness Analysis

This repository contains all of the code and data required to reproduce the experiments presented in our ophthalmology note generation study.  The project evaluates how adversarial prompts affect the quality of GPT‑generated SOAP notes by comparing clean and adversarial outputs at both the token and entity levels.  All dependencies on Google Drive and Colab have been removed; everything runs locally from the repository.

## Repository structure

```
project_repo/
├── data/                   # Input data files (case studies and generated notes)
│   ├── clean_case_studies.docx              # Example clean case study document
│   ├── adversarial_case_studies.docx        # Example adversarial case study document
│   ├── 1_CLEAN_soap_notes.txt               # Clean SOAP notes for pair 1
│   ├── 1_ADV_soap_notes.txt                 # Adversarial SOAP notes for pair 1
│   ├── 2_CLEAN_soap_notes.txt               # Clean SOAP notes for pair 2
│   └── 2_ADV_soap_notes.txt                 # Adversarial SOAP notes for pair 2
├── notebooks/              # Jupyter notebooks for each stage of the analysis
│   ├── generate_soap_notes.ipynb       # Generates SOAP notes from case studies using GPT‑5
│   ├── extract_entities.ipynb          # Runs Bio_ClinicalBERT NER on generated notes
│   ├── bertscore_similarity.ipynb      # Computes BERTScore similarity between note pairs
│   └── permutation_test.ipynb          # Performs a simple permutation test on similarity metrics
├── results/                # Output tables and intermediate results
│   ├── ner_results.csv
│   ├── bertscore_results.csv
│   └── local_ner_results.csv           # Produced by `extract_entities.ipynb`
├── src/
│   └── pipeline_code.py    # Script for computing entity‑level metrics and permutation tests
├── paper/                  # Research paper and figures
│   └── updated_paper.docx
├── config_template.py      # Template for API keys and model configuration
└── README.md               # This file
```

## Setup

1. **Clone the repository** and install the Python dependencies listed in `requirements.txt` (e.g., via `pip install -r requirements.txt`).  Make sure you have access to a GPU if you plan to run the note generation notebook, as GPT‑5 models can be resource‑intensive.
2. **Add your API keys**.  Copy `config_template.py` to `config.py` and fill in your `DARTMOUTH_API_KEY`, `DARTMOUTH_CHAT_API_KEY`, and `SELECTED_MODEL`.  These keys are required for the GPT‑5 note generation.
3. **Place your case study documents** in the `data/` folder.  This example repository includes an `adversarial_case_studies.docx` file for demonstration.  Replace or add your own case study documents as needed.

## Pipeline overview

1. **Generate SOAP notes** – Run `notebooks/generate_soap_notes.ipynb`.  This notebook reads the clean and adversarial case studies from `data/`, uses your API keys to call the GPT‑5 model, and writes out clean/adversarial SOAP note files into the same directory.
2. **Extract entities** – Run `notebooks/extract_entities.ipynb`.  It loads each `.txt` note from `data/`, runs the Bio_ClinicalBERT NER model to extract medical entities, and saves the results to `results/local_ner_results.csv`.
3. **Compute BERTScore similarities** – Run `notebooks/bertscore_similarity.ipynb`.  It pairs the clean and adversarial notes, computes BERTScore precision/recall/F1 using a clinical BERT model, and saves the scores to `results/bertscore_results.csv`.
4. **Run permutation test** – Run `notebooks/permutation_test.ipynb` to perform a simple permutation test on the BERTScore F1 scores.  The notebook reports the observed mean F1 and a p‑value indicating whether the matched pairs are more similar than random pairings.
5. **Entity‑level analysis** – Execute `src/pipeline_code.py` from the command line to compute additional metrics, including entity‑level Jaccard similarity, counts of missing/added entities, TF‑IDF cosine similarity, and a permutation test on these metrics:

   ```bash
   python src/pipeline_code.py --notes_dir data --ner_file results/local_ner_results.csv --output_dir results
   ```

   The script writes summary tables to the `results/` directory.

## Notes

* The notebooks and scripts assume that the clean and adversarial notes follow the naming pattern `X_CLEAN_soap_notes.txt` and `X_ADV_soap_notes.txt`, where `X` is the pair number (e.g. `1_CLEAN_soap_notes.txt`).  If you add more pairs, update the `note_pairs` list in `bertscore_similarity.ipynb` accordingly.
* API keys are kept outside of the notebooks to avoid accidentally committing secrets.  Do **not** commit your `config.py` to version control.
* This repository is designed to be self‑contained; there is no reliance on Google Drive, Colab, or external downloads once your API keys and case studies are provided.
