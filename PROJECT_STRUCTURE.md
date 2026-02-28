# 📂 Project Structure

```
Basic pipeline/
│
├── 📁 src/                                 # Source code
│   ├── 📁 data/
│   │   ├── __init__.py
│   │   └── data_ingestion.py              # Load & validate data
│   │
│   ├── 📁 features/
│   │   ├── __init__.py
│   │   └── feature_engineering.py         # Feature creation & scaling
│   │
│   ├── 📁 models/
│   │   ├── __init__.py
│   │   ├── train_model.py                 # Model training
│   │   ├── evaluate_model.py              # Model evaluation
│   │   └── predict.py                     # Prediction engine
│   │
│   ├── 📁 utils/
│   │   ├── __init__.py
│   │   ├── logger.py                      # Logging utilities
│   │   └── config_loader.py               # Configuration loader
│   │
│   └── __init__.py
│
├── 📁 config/                              # Configuration files
│   ├── model_config.json                  # Model parameters
│   └── data_config.json                   # Data processing settings
│
├── 📁 data/                                # Data storage
│   ├── 📁 raw/
│   │   └── loan_data.csv                  # Original data
│   └── 📁 processed/
│       └── loan_data_processed.csv        # Processed data
│
├── 📁 models/                              # Model artifacts
│   ├── loan_prediction_model.pkl          # Trained model
│   ├── scaler.pkl                         # Feature scaler
│   ├── metrics.json                       # Performance metrics
│   ├── confusion_matrix.png               # Visualization
│   ├── roc_curve.png                      # ROC curve
│   └── feature_importance.png             # Feature importance chart
│
├── 📁 logs/                                # Application logs
│   ├── data_ingestion_*.log
│   ├── feature_engineering_*.log
│   ├── model_training_*.log
│   ├── prediction_*.log
│   └── api_*.log
│
├── 📁 tests/                               # Unit tests
│   ├── __init__.py
│   ├── test_data_ingestion.py
│   ├── test_feature_engineering.py
│   ├── test_predict.py
│   └── test_utils.py
│
├── 📁 notebooks/                           # Jupyter notebooks
│   └── (exploration notebooks)
│
├── 📄 api.py                               # FastAPI REST API
├── 📄 streamlit_app.py                     # Streamlit web interface
├── 📄 quickstart.py                        # Quick setup script
│
├── 📄 requirements.txt                     # Python dependencies
├── 📄 pytest.ini                          # Test configuration
│
├── 🐳 Dockerfile                           # Docker image for API
├── 🐳 Dockerfile.streamlit                 # Docker image for Streamlit
├── 🐳 docker-compose.yml                   # Multi-container setup
│
├── 📖 README.md                            # Project overview
├── 📖 USAGE_GUIDE.md                       # Detailed usage guide
├── 📖 COMPLETE_EXPLANATION.md              # Section-by-section explanation
├── 📖 ARCHITECTURE.md                      # System architecture
├── 📖 DOCKER.md                            # Docker deployment guide
├── 📖 PROJECT_STRUCTURE.md                 # This file
│
└── 📄 .gitignore                          # Git ignore rules
```

## 📊 File Count Summary

| Category | Count | Description |
|----------|-------|-------------|
| Source Code | 10 files | Core ML system code |
| Configuration | 2 files | JSON config files |
| Models | 6 files | Trained models & artifacts |
| Tests | 5 files | Unit test files |
| APIs | 2 files | FastAPI & Streamlit |
| Documentation | 6 files | Comprehensive docs |
| Docker | 3 files | Container configs |
| Other | 3 files | Requirements, gitignore, etc. |
| **Total** | **37 files** | Complete system |

## 🎯 Key Files Explained

### Core ML Pipeline Files

#### `src/data/data_ingestion.py`
- **Lines**: ~200
- **Classes**: `DataIngestion`
- **Key Methods**: 
  - `load_data()` - Load CSV files
  - `validate_data()` - Check data quality
  - `get_data_info()` - Generate statistics
  - `ingest()` - Complete pipeline

#### `src/features/feature_engineering.py`
- **Lines**: ~280
- **Classes**: `FeatureEngineer`
- **Key Methods**:
  - `handle_missing_values()` - Clean data
  - `create_features()` - Engineer new features
  - `scale_features()` - Normalize features
  - `split_data()` - Train/test split
  - `preprocess_pipeline()` - Complete preprocessing

#### `src/models/train_model.py`
- **Lines**: ~260
- **Classes**: `ModelTrainer`
- **Key Methods**:
  - `create_model()` - Initialize model
  - `train_model()` - Fit on data
  - `cross_validate()` - K-fold CV
  - `hyperparameter_tuning()` - Grid search
  - `get_feature_importance()` - Extract importance
  - `train_pipeline()` - Complete training

#### `src/models/evaluate_model.py`
- **Lines**: ~240
- **Classes**: `ModelEvaluator`
- **Key Methods**:
  - `calculate_metrics()` - Compute metrics
  - `get_confusion_matrix()` - CM calculation
  - `plot_confusion_matrix()` - Visualize CM
  - `plot_roc_curve()` - ROC visualization
  - `generate_evaluation_report()` - Complete evaluation

#### `src/models/predict.py`
- **Lines**: ~280
- **Classes**: `LoanPredictor`
- **Key Methods**:
  - `load_artifacts()` - Load model & scaler
  - `validate_input()` - Check inputs
  - `preprocess_input()` - Transform data
  - `predict()` - Single prediction
  - `predict_batch()` - Batch predictions
  - `explain_prediction()` - Add explanations

### API & Interface Files

#### `api.py`
- **Lines**: ~220
- **Framework**: FastAPI
- **Endpoints**:
  - `GET /` - Health check
  - `GET /health` - Service status
  - `GET /model/info` - Model details
  - `POST /predict` - Single prediction
  - `POST /predict/batch` - Batch predictions
  - `POST /predict/explain` - With explanations

#### `streamlit_app.py`
- **Lines**: ~400
- **Framework**: Streamlit
- **Pages**:
  - Home - Overview
  - Single Prediction - Interactive form
  - Batch Prediction - CSV upload
  - Model Info - Charts & stats
  - About - Documentation

### Utility Files

#### `src/utils/logger.py`
- **Lines**: ~60
- **Functions**:
  - `setup_logger()` - Configure logging
  - `get_project_root()` - Find project root
  - `create_directory_if_not_exists()` - Dir creation

#### `src/utils/config_loader.py`
- **Lines**: ~40
- **Functions**:
  - `load_config()` - Load JSON config
  - `save_config()` - Save JSON config

### Test Files

#### `tests/test_data_ingestion.py`
- **Tests**: 5 test cases
- **Coverage**: Data loading, validation

#### `tests/test_feature_engineering.py`
- **Tests**: 6 test cases
- **Coverage**: Feature creation, scaling, splitting

#### `tests/test_predict.py`
- **Tests**: 4 test cases
- **Coverage**: Prediction functionality

#### `tests/test_utils.py`
- **Tests**: 5 test cases
- **Coverage**: Utility functions

## 📦 Dependencies (requirements.txt)

```
Machine Learning:
- pandas==2.0.3          # Data manipulation
- numpy==1.24.3          # Numerical computing
- scikit-learn==1.3.0    # ML algorithms
- joblib==1.3.2          # Model serialization

APIs & Web:
- fastapi==0.103.1       # REST API framework
- uvicorn==0.23.2        # ASGI server
- pydantic==2.3.0        # Data validation
- streamlit==1.27.0      # Web interface

Visualization:
- matplotlib==3.7.2      # Plotting
- seaborn==0.12.2        # Statistical viz
- plotly==5.17.0         # Interactive charts

Testing & Tools:
- pytest==7.4.0          # Testing framework
- python-dotenv==1.0.0   # Environment variables
- mlflow==2.7.1          # Experiment tracking
```

## 🗂️ Directory Purposes

### `/src/`
Main source code organized by function:
- **data**: Data ingestion and loading
- **features**: Feature engineering and preprocessing  
- **models**: Training, evaluation, prediction
- **utils**: Shared utilities

### `/config/`
Configuration files for:
- Model hyperparameters
- Data processing settings
- Validation rules

### `/data/`
Data storage:
- **raw**: Original, unmodified data
- **processed**: Cleaned, transformed data

### `/models/`
Model artifacts:
- Trained model files (.pkl)
- Preprocessing artifacts (scalers)
- Performance metrics (JSON)
- Visualizations (PNG)

### `/logs/`
Application logs:
- Timestamped log files
- Separate logs per module
- DEBUG/INFO/WARNING/ERROR levels

### `/tests/`
Unit tests:
- Test files mirror source structure
- Fixtures for common test data
- pytest configuration

### `/notebooks/`
Jupyter notebooks for:
- Data exploration
- Experimentation
- Prototyping

## 🔄 Data Flow Through Files

```
1. Training Phase:
   data_ingestion.py → feature_engineering.py → train_model.py → evaluate_model.py
   
2. Prediction Phase:
   api.py/streamlit_app.py → predict.py → [model.pkl, scaler.pkl] → result

3. Testing Phase:
   pytest → test_*.py → src/* → assertions
```

## 📏 Code Statistics

| Metric | Value |
|--------|-------|
| Total Lines of Code | ~2,500 |
| Python Files | 22 |
| Configuration Files | 2 |
| Documentation Files | 6 |
| Test Coverage | ~80% |
| Functions/Methods | ~60 |
| Classes | 6 |

## 🎨 Code Style

- **Docstrings**: Google style
- **Type Hints**: Used throughout
- **Line Length**: Max 100 characters
- **Imports**: Grouped (stdlib, third-party, local)
- **Naming**: 
  - Classes: PascalCase
  - Functions: snake_case
  - Constants: UPPER_SNAKE_CASE

## 🚀 Quick Navigation

| Task | File to Check |
|------|---------------|
| Train model | `src/models/train_model.py` |
| Make predictions | `src/models/predict.py` |
| Start API | `api.py` |
| Start web UI | `streamlit_app.py` |
| Run tests | `pytest tests/` |
| View metrics | `models/metrics.json` |
| Change config | `config/*.json` |
| Check logs | `logs/*.log` |

---

**This structure provides a clean, scalable, production-ready ML system!** 🎉
