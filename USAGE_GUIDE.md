# Complete ML System Usage Guide

## Overview

This is a production-ready, end-to-end machine learning system for loan prediction. Below is a comprehensive explanation of each section and how to use the system.

---

## 📁 System Architecture

```
Basic pipeline/
├── src/                    # Source code
│   ├── data/              # Data ingestion modules
│   ├── features/          # Feature engineering
│   ├── models/            # Model training, evaluation, prediction
│   └── utils/             # Utilities (logging, config)
├── config/                # Configuration files
├── data/                  # Data storage
├── models/                # Trained model artifacts
├── tests/                 # Unit tests
├── notebooks/             # Jupyter notebooks
├── logs/                  # Application logs
├── api.py                 # REST API
├── streamlit_app.py       # Web interface
└── quickstart.py          # Quick start script
```

---

## 🚀 Getting Started

### 1. Installation

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Quick Start

Run the automated setup:

```bash
python quickstart.py
```

This will:
- Train the model
- Evaluate performance
- Make sample predictions
- Show you next steps

---

## 📊 System Components Explained

### 1. Data Ingestion (`src/data/data_ingestion.py`)

**Purpose**: Load and validate raw data

**Key Features**:
- Loads data from CSV files
- Validates data schema and quality
- Checks for missing values and data types
- Provides data statistics

**Usage**:
```python
from src.data.data_ingestion import DataIngestion

ingestion = DataIngestion()
df, info = ingestion.ingest()
print(f"Loaded {info['shape'][0]} records")
```

**What it does**:
1. Reads data from `data/raw/loan_data.csv`
2. Validates required columns exist
3. Checks age, income, savings are in valid ranges
4. Reports missing values and duplicates
5. Returns clean DataFrame with metadata

---

### 2. Feature Engineering (`src/features/feature_engineering.py`)

**Purpose**: Transform raw data into ML-ready features

**Key Features**:
- Handles missing values
- Creates engineered features
- Scales features using StandardScaler
- Splits data into train/test sets

**Usage**:
```python
from src.features.feature_engineering import FeatureEngineer

engineer = FeatureEngineer()
X_train, X_test, y_train, y_test = engineer.preprocess_pipeline(df)
```

**Feature Engineering Steps**:
1. **Handle Missing Values**: Drop or impute missing data
2. **Create New Features**:
   - `savings_to_income_ratio` = savings / income
   - `age_group_encoded` = categorical age groups (young, mid, senior, elderly)
   - `income_category_encoded` = income brackets (low, medium, high, very_high)
3. **Scale Features**: Standardize using StandardScaler (mean=0, std=1)
4. **Split Data**: 80% train, 20% test with stratification

---

### 3. Model Training (`src/models/train_model.py`)

**Purpose**: Train machine learning models

**Key Features**:
- Logistic Regression model
- Cross-validation
- Hyperparameter tuning
- Feature importance analysis
- Model serialization

**Usage**:
```python
from src.models.train_model import ModelTrainer

trainer = ModelTrainer()
results, X_train, X_test, y_train, y_test = trainer.train_pipeline()
```

**Training Process**:
1. **Data Loading**: Ingests data using DataIngestion
2. **Preprocessing**: Applies feature engineering
3. **Model Training**: Trains Logistic Regression
4. **Cross-Validation**: 5-fold CV for robust evaluation
5. **Feature Importance**: Extracts model coefficients
6. **Model Saving**: Saves to `models/loan_prediction_model.pkl`

**With Hyperparameter Tuning**:
```python
results = trainer.train_pipeline(perform_tuning=True)
```
This tests different C values and solvers to find optimal parameters.

---

### 4. Model Evaluation (`src/models/evaluate_model.py`)

**Purpose**: Comprehensive model performance analysis

**Key Features**:
- Multiple performance metrics
- Confusion matrix visualization
- ROC curve plotting
- Classification reports
- Metric persistence

**Usage**:
```python
from src.models.evaluate_model import ModelEvaluator

evaluator = ModelEvaluator()
results = evaluator.generate_evaluation_report(model, X_test, y_test)
```

**Evaluation Metrics**:
- **Accuracy**: Overall correctness (TP+TN)/(Total)
- **Precision**: Of predicted approvals, how many were correct
- **Recall**: Of actual approvals, how many were caught
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve (discrimination ability)

**Visualizations Created**:
- `models/confusion_matrix.png` - Visual confusion matrix
- `models/roc_curve.png` - ROC curve with AUC score
- `models/feature_importance.png` - Feature contribution chart

---

### 5. Prediction/Inference (`src/models/predict.py`)

**Purpose**: Make predictions on new data

**Key Features**:
- Single predictions
- Batch predictions
- Input validation
- Probability scores
- Prediction explanations

**Usage**:

**Single Prediction**:
```python
from src.models.predict import LoanPredictor

predictor = LoanPredictor()
result = predictor.predict(age=35, income=50000, savings=15000)

print(f"Decision: {result['approval_status']}")
print(f"Probability: {result['probability']:.2%}")
```

**Batch Prediction**:
```python
batch_data = [
    {'age': 35, 'income': 50000, 'savings': 15000},
    {'age': 25, 'income': 30000, 'savings': 5000}
]

results = predictor.predict_batch(batch_data)
```

**With Explanation**:
```python
explanation = predictor.explain_prediction(age=30, income=35000, savings=5000)
print("Factors:", explanation['factors'])
```

---

### 6. REST API (`api.py`)

**Purpose**: Serve predictions via HTTP endpoints

**Key Features**:
- FastAPI framework
- RESTful endpoints
- Auto-generated documentation
- Input validation
- Error handling

**Starting the API**:
```bash
python api.py
```

API runs at: `http://localhost:8000`
Docs available at: `http://localhost:8000/docs`

**Endpoints**:

**GET /** - Health check
```bash
curl http://localhost:8000/
```

**POST /predict** - Single prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"age": 35, "income": 50000, "savings": 15000}'
```

**POST /predict/batch** - Batch predictions
```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{"applications": [{"age": 35, "income": 50000, "savings": 15000}]}'
```

**POST /predict/explain** - Prediction with explanation
```bash
curl -X POST "http://localhost:8000/predict/explain" \
  -H "Content-Type: application/json" \
  -d '{"age": 30, "income": 35000, "savings": 5000}'
```

---

### 7. Web Interface (`streamlit_app.py`)

**Purpose**: User-friendly web UI for predictions

**Key Features**:
- Interactive forms
- Single and batch predictions
- Visual dashboards
- Model information display
- CSV upload/download

**Starting the Web UI**:
```bash
streamlit run streamlit_app.py
```

Access at: `http://localhost:8501`

**Pages**:
1. **Home**: Overview and quick start
2. **Single Prediction**: Enter details, get instant prediction
3. **Batch Prediction**: Upload CSV, get results for all
4. **Model Info**: View model details and feature importance
5. **About**: System information and documentation

---

## 🔧 Configuration

### Model Configuration (`config/model_config.json`)

```json
{
  "model": {
    "name": "LogisticRegression",
    "random_state": 42,
    "max_iter": 1000
  },
  "data_split": {
    "test_size": 0.2,
    "random_state": 42
  }
}
```

### Data Configuration (`config/data_config.json`)

```json
{
  "data_source": {
    "raw_data_path": "data/raw/loan_data.csv"
  },
  "validation": {
    "age_range": [18, 100],
    "income_range": [0, 1000000]
  }
}
```

---

## 🧪 Testing

Run all tests:
```bash
pytest tests/ -v
```

Run specific test:
```bash
pytest tests/test_predict.py -v
```

**Test Coverage**:
- Data ingestion validation
- Feature engineering correctness
- Model prediction accuracy
- API endpoint responses
- Utility function behavior

---

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)

```bash
# Build and start all services
docker-compose up --build

# Stop services
docker-compose down
```

**Services**:
- API: http://localhost:8000
- Streamlit: http://localhost:8501

### Individual Containers

**API Only**:
```bash
docker build -t loan-api .
docker run -p 8000:8000 -v $(pwd)/models:/app/models loan-api
```

**Streamlit Only**:
```bash
docker build -f Dockerfile.streamlit -t loan-streamlit .
docker run -p 8501:8501 -v $(pwd)/models:/app/models loan-streamlit
```

---

## 📝 Logging

All components use structured logging:

**Log Files**:
- `logs/data_ingestion_*.log` - Data loading logs
- `logs/feature_engineering_*.log` - Preprocessing logs
- `logs/model_training_*.log` - Training logs
- `logs/prediction_*.log` - Prediction logs
- `logs/api_*.log` - API request logs

**Log Format**:
```
2026-02-28 10:30:00 - module_name - INFO - Message
```

---

## 🔄 Complete Workflow

### Development Workflow

1. **Train Model**:
```bash
python -m src.models.train_model
```

2. **Evaluate Model**:
```bash
python -m src.models.evaluate_model
```

3. **Test Predictions**:
```bash
python -m src.models.predict
```

4. **Run Tests**:
```bash
pytest tests/ -v
```

5. **Start API**:
```bash
python api.py
```

6. **Start Web UI**:
```bash
streamlit run streamlit_app.py
```

### Production Deployment

1. **Train and validate model**
2. **Run full test suite**
3. **Build Docker images**
4. **Deploy with docker-compose**
5. **Monitor logs for issues**
6. **Set up health checks**

---

## 🎯 Use Cases

### Use Case 1: Real-time Loan Approval

1. Customer applies online
2. Application sent to API endpoint
3. Model predicts approval probability
4. Instant decision returned
5. Application logged for audit

### Use Case 2: Batch Processing

1. Upload daily applications CSV
2. Process all applications at once
3. Generate approval report
4. Export results for review
5. Send notifications to applicants

### Use Case 3: Manual Review

1. Loan officer uses web interface
2. Enters applicant details
3. Views prediction and explanation
4. Uses insights for final decision
5. Logs decision with reasoning

---

## 📊 Model Performance

**Expected Performance**:
- Accuracy: ~85%
- Precision: ~83%
- Recall: ~87%
- F1-Score: ~85%
- ROC-AUC: ~0.90

**Feature Importance** (typical):
1. Income (highest impact)
2. Savings
3. Age
4. Savings-to-income ratio
5. Income category

---

## 🛠️ Troubleshooting

### Model not found error
```bash
# Train the model first
python -m src.models.train_model
```

### Import errors
```bash
# Make sure you're in the project root
cd "C:\Users\acer\Desktop\Learning projects\Basic pipeline"

# Reinstall dependencies
pip install -r requirements.txt
```

### API won't start
```bash
# Check if port 8000 is available
netstat -ano | findstr :8000

# Use different port
uvicorn api:app --port 8080
```

### Streamlit issues
```bash
# Clear cache
streamlit cache clear

# Run with different port
streamlit run streamlit_app.py --server.port 8502
```

---

## 📚 Additional Resources

- **API Documentation**: http://localhost:8000/docs
- **ReDoc API Docs**: http://localhost:8000/redoc
- **Docker Guide**: See DOCKER.md
- **README**: See README.md

---

## 🔐 Security Considerations

1. **Input Validation**: All inputs validated before processing
2. **Error Handling**: Graceful error handling with informative messages
3. **Logging**: All operations logged for audit
4. **Rate Limiting**: Consider adding to API for production
5. **Authentication**: Add if deploying publicly

---

## 🚀 Future Enhancements

- Additional ML models (Random Forest, XGBoost)
- Model drift detection
- A/B testing framework
- Real-time monitoring dashboard
- Database integration
- User authentication
- Email notifications
- Advanced analytics

---

**End of Guide**
