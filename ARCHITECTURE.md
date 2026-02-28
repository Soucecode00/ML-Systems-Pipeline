# System Architecture

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     LOAN PREDICTION SYSTEM                       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐  │
│  │  Raw Data    │ ───> │  Validation  │ ───> │  Processed   │  │
│  │  (CSV)       │      │  & Cleaning  │      │  Data        │  │
│  └──────────────┘      └──────────────┘      └──────────────┘  │
│         │                     │                      │           │
│         └─────────────────────┴──────────────────────┘           │
│                              │                                   │
│                    DataIngestion Module                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FEATURE ENGINEERING LAYER                     │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐  │
│  │  Missing     │      │   Feature    │      │   Feature    │  │
│  │  Value       │ ───> │   Creation   │ ───> │   Scaling    │  │
│  │  Handling    │      │              │      │              │  │
│  └──────────────┘      └──────────────┘      └──────────────┘  │
│         │                     │                      │           │
│         └─────────────────────┴──────────────────────┘           │
│                              │                                   │
│                   FeatureEngineer Module                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      MODEL LAYER                                 │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐  │
│  │   Model      │      │     Cross    │      │    Model     │  │
│  │   Training   │ ───> │  Validation  │ ───> │  Evaluation  │  │
│  │              │      │              │      │              │  │
│  └──────────────┘      └──────────────┘      └──────────────┘  │
│         │                     │                      │           │
│         └─────────────────────┴──────────────────────┘           │
│                              │                                   │
│              ModelTrainer & ModelEvaluator Modules               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PREDICTION LAYER                              │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐  │
│  │   Load       │      │  Preprocess  │      │   Predict    │  │
│  │   Model      │ ───> │    Input     │ ───> │   & Score    │  │
│  │              │      │              │      │              │  │
│  └──────────────┘      └──────────────┘      └──────────────┘  │
│         │                     │                      │           │
│         └─────────────────────┴──────────────────────┘           │
│                              │                                   │
│                    LoanPredictor Module                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                 ┌────────────┴────────────┐
                 ▼                         ▼
┌──────────────────────────┐  ┌──────────────────────────┐
│     SERVING LAYER        │  │     UI LAYER             │
├──────────────────────────┤  ├──────────────────────────┤
│  ┌────────────────────┐  │  │  ┌────────────────────┐  │
│  │   FastAPI REST     │  │  │  │   Streamlit Web    │  │
│  │       API          │  │  │  │     Interface      │  │
│  │                    │  │  │  │                    │  │
│  │ • /predict         │  │  │  │ • Single Predict   │  │
│  │ • /predict/batch   │  │  │  │ • Batch Predict    │  │
│  │ • /predict/explain │  │  │  │ • Visualizations   │  │
│  │ • /health          │  │  │  │ • Model Info       │  │
│  │ • /model/info      │  │  │  │                    │  │
│  └────────────────────┘  │  │  └────────────────────┘  │
│           │              │  │           │              │
│      Port 8000           │  │      Port 8501           │
└──────────────────────────┘  └──────────────────────────┘
                 │                         │
                 └────────────┬────────────┘
                              ▼
                    ┌──────────────────┐
                    │      Users       │
                    │  • Developers    │
                    │  • Loan Officers │
                    │  • Applications  │
                    └──────────────────┘
```

## Component Details

### 1. Data Layer
- **Input**: CSV files with loan application data
- **Processing**: Validation, cleaning, quality checks
- **Output**: Clean, validated DataFrame
- **Module**: `src/data/data_ingestion.py`

### 2. Feature Engineering Layer
- **Input**: Raw DataFrame
- **Processing**: 
  - Missing value handling
  - Feature creation (ratios, categories)
  - Standardization/Scaling
- **Output**: ML-ready feature matrix
- **Module**: `src/features/feature_engineering.py`

### 3. Model Layer
- **Training**: Logistic Regression with scikit-learn
- **Validation**: 5-fold cross-validation
- **Evaluation**: Multiple metrics, visualizations
- **Modules**: 
  - `src/models/train_model.py`
  - `src/models/evaluate_model.py`

### 4. Prediction Layer
- **Input**: New application data
- **Processing**: Feature engineering + model inference
- **Output**: Approval decision + probability
- **Module**: `src/models/predict.py`

### 5. Serving Layer (API)
- **Framework**: FastAPI
- **Features**: RESTful endpoints, auto docs
- **Endpoints**: Predict, batch predict, explain
- **File**: `api.py`

### 6. UI Layer
- **Framework**: Streamlit
- **Features**: Interactive forms, visualizations
- **Pages**: Single/batch prediction, model info
- **File**: `streamlit_app.py`

## Data Flow

### Training Flow
```
Raw CSV → Ingestion → Validation → Feature Engineering → 
Train/Test Split → Model Training → Cross Validation → 
Evaluation → Save Model & Metrics
```

### Prediction Flow
```
New Data → Validation → Feature Engineering → 
Load Model → Predict → Return Result
```

## Technology Stack

| Layer | Technology |
|-------|-----------|
| ML Framework | scikit-learn |
| API Framework | FastAPI |
| Web UI | Streamlit |
| Data Processing | pandas, numpy |
| Visualization | matplotlib, seaborn, plotly |
| Testing | pytest |
| Logging | Python logging |
| Configuration | JSON |
| Containerization | Docker |

## Deployment Options

### Option 1: Local Development
```bash
python api.py
streamlit run streamlit_app.py
```

### Option 2: Docker Compose
```bash
docker-compose up --build
```

### Option 3: Individual Services
```bash
# API
docker run -p 8000:8000 loan-api

# Streamlit
docker run -p 8501:8501 loan-streamlit
```

## Scalability Considerations

### Current Architecture
- Single model serving
- In-memory predictions
- File-based model storage

### Production Enhancements
1. **Load Balancing**: Multiple API instances
2. **Caching**: Redis for frequent predictions
3. **Database**: PostgreSQL for audit logs
4. **Message Queue**: RabbitMQ for async processing
5. **Model Registry**: MLflow for version control
6. **Monitoring**: Prometheus + Grafana

## Security Features

1. **Input Validation**: Pydantic schemas
2. **Error Handling**: Graceful degradation
3. **Logging**: Comprehensive audit trail
4. **CORS**: Configurable origins
5. **Rate Limiting**: (To be added in production)

## Monitoring & Observability

### Logs
- Application logs in `logs/`
- Structured logging format
- Different log levels (INFO, WARNING, ERROR)

### Metrics
- Model performance metrics saved
- Prediction counts (can be tracked)
- API response times (via FastAPI)

### Health Checks
- API: `/health` endpoint
- Model: Load verification on startup
- Data: Validation checks

## Configuration Management

### Config Files
- `config/model_config.json` - Model parameters
- `config/data_config.json` - Data processing settings

### Environment Variables
- Can be set via `.env` file
- Docker environment variables
- OS environment variables
