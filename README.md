# Data-Agnostic MLOps Pipeline

> ⚠️ **Note:** This project is currently under active development. Some features may be incomplete or subject to change.

A modular and extensible MLOps pipeline that automatically handles different data formats and performs end-to-end data processing and model training.

---

## 🚀 Features

- **Data-Agnostic Ingestion** – Supports CSV, Excel (.xlsx/.xls), and JSON files
- **Auto Preprocessing** – Automatically detects column types and handles:
  - Missing value imputation (mean for numeric, mode for categorical)
  - Label encoding for categorical columns
- **Modular Architecture** – Clean separation of concerns with dedicated modules
- **Logging** – Professional logging with timestamps for debugging

---

## 📁 Project Structure

```
Data-agnostic-MLOps-pipeline/
├── main.py                    # Entry point - runs the data pipeline
├── requirements.txt           # Python dependencies
├── configs/
│   └── params.yaml            # Configuration parameters
├── data/
│   ├── raw/                   # Input data files
│   └── processed/             # Cleaned output data
├── pipelines/
│   ├── train_pipeline.py      # Training pipeline orchestration
│   └── inference_pipeline.py  # Inference pipeline (WIP)
├── src/
│   ├── ingestion/
│   │   └── ingest_data.py     # Data loading module
│   ├── preprocessing/
│   │   └── preprocess.py      # Data cleaning & encoding
│   ├── training/
│   │   └── train.py           # Model training (Logistic Regression)
│   ├── validation/
│   │   └── validate_data.py   # Data validation checks
│   ├── evaluation/
│   ├── deployment/
│   ├── monitoring/
│   ├── registry/
│   └── utils/
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
└── tests/
    └── test_pipeline.py
```

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Data-agnostic-MLOps-pipeline.git
cd Data-agnostic-MLOps-pipeline

# Install dependencies
pip install -r requirements.txt
```

---

## 🏃 Usage

### Run Data Pipeline

```bash
python main.py
```

This will:
1. Load raw data from `data/raw/`
2. Clean and preprocess the data
3. Save processed data to `data/processed/clean_data.csv`

---

## 📦 Dependencies

- pandas
- numpy
- scikit-learn
- openpyxl

---

## 🔧 Modules

| Module | Description |
|--------|-------------|
| `ingest_data.py` | Loads data from CSV/Excel/JSON based on file extension |
| `preprocess.py` | Handles missing values and encodes categorical columns |
| `train.py` | Trains Logistic Regression model |
| `validate_data.py` | Checks for missing values in dataset |

---

## 📝 License

This project is open source and available under the MIT License.