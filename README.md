# Phishing Detection - Data Science in Cyber Final Project

This repository contains the implementation and report for a Data Science in Cyber final project on phishing detection.

## Project topic

Phishing Detection using machine learning on URL, domain, HTML, and JavaScript indicators.

## Selected source

- Selected tutorial/repository: `shreyagopal/Phishing-Website-Detection-by-Machine-Learning-Techniques`
- Repository: https://github.com/shreyagopal/Phishing-Website-Detection-by-Machine-Learning-Techniques
- Original data file: `DataFiles/5.urldata.csv`
- Original model-training notebook: `Phishing Website Detection_Models & Training.ipynb`
- Original feature-extraction notebook: `URL Feature Extraction.ipynb`

## Dataset source

The selected source states that phishing URLs were sampled from PhishTank and legitimate URLs from the University of New Brunswick URL dataset. The final extracted dataset contains 10,000 rows, 17 extracted URL/domain/HTML/JavaScript features, and a binary label where `1` means phishing and `0` means legitimate.

The local dataset copy used by the notebook is:

```text
data/5.urldata.csv
```

## Repository files

```text
phishing_detection_report.pdf        # Final report
phishing_detection_project.ipynb     # Executable notebook
README.md                            # Repository documentation
requirements.txt                     # Python dependencies
data/5.urldata.csv                   # Dataset used by the notebook
data/phishing_processed_features.csv # Processed data created by the analysis
figures/                             # Generated visualizations
outputs/                             # Generated metrics, summaries, and error examples
run_analysis.py                      # Supporting script that regenerates analysis outputs
```

## Reproducibility and execution instructions

1. Clone this repository or download the project folder.
2. Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\\Scripts\\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the notebook:

```bash
jupyter notebook phishing_detection_project.ipynb
```

5. Execute all notebook cells from top to bottom. The notebook regenerates the files in `figures/` and `outputs/`.

Optional: regenerate all analysis outputs from the supporting script:

```bash
python run_analysis.py
```

## Reproducibility notes

- Fixed random seed: 42.
- The dataset is included locally, so the project runs without downloading data.
- The analysis reports both a random stratified 80/20 split and a grouped split by domain.
- The grouped split is included because the data contains duplicated rows and duplicated domains, which can make random-split results optimistic.
- The report and notebook focus on critical evaluation of the selected source, not only reproduction.
