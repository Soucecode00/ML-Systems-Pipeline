# Loan Prediction ML System

A complete end-to-end machine learning system for predicting loan approvals with a production-ready architecture.

## 🎯 Project Overview

This project demonstrates a professional ML pipeline for loan prediction, including:
- Data ingestion and preprocessing
- Feature engineering
- Model training with experiment tracking
- Model evaluation and validation
- REST API for predictions
- Web interface for user interaction
- Logging and monitoring
- Containerization for deployment

## 📁 Project Structure

```
Basic pipeline/
├── src/
│   ├── data/           # Data ingestion and loading
│   ├── features/       # Feature engineering and preprocessing
│   ├── models/         # Model training and evaluation
│   └── utils/          # Utility functions (logging, config)
├── config/             # Configuration files
├── data/
│   ├── raw/           # Raw data files
│   └── processed/     # Processed data files
├── models/            # Saved model artifacts
├── notebooks/         # Jupyter notebooks for exploration
├── tests/             # Unit tests
├── logs/              # Application logs
├── api.py             # FastAPI REST API
├── streamlit_app.py   # Streamlit web interface
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## 🚀 Features

### 1. Data Pipeline
- Automated data ingestion from CSV files
- Data validation and quality checks
- Feature preprocessing and scaling
- Train-test splitting with stratification

### 2. Model Training
- Logistic Regression baseline model
- Hyperparameter tuning
- Cross-validation
- Experiment tracking with MLflow
- Model versioning and artifact storage

### 3. Model Evaluation
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix visualization
- ROC-AUC curve
- Classification reports

### 4. Deployment
- FastAPI REST API for predictions
- Streamlit web interface
- Docker containerization
- Structured logging

## 🛠️ Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Usage

### Training the Model

```bash
python -m src.models.train_model
```

### Running the API Server

```bash
python api.py
```

Access the API documentation at: `http://localhost:8000/docs`

### Running the Web Interface

```bash
streamlit run streamlit_app.py
```

### Making Predictions

```python
from src.models.predict import LoanPredictor

predictor = LoanPredictor()
result = predictor.predict(age=35, income=50000, savings=15000)
print(f"Approved: {result['approved']}, Probability: {result['probability']:.2%}")
```

## 🧪 Testing

Run unit tests:
```bash
pytest tests/
```

## 📈 Model Performance

The current model achieves:
- Accuracy: ~85%
- Precision: ~83%
- Recall: ~87%
- F1-Score: ~85%

## 🔧 Configuration

Configuration files are stored in `config/` directory:
- `model_config.json`: Model hyperparameters
- `data_config.json`: Data processing settings

## 📝 API Endpoints

### POST /predict
Predict loan approval for a single application

**Request:**
```json
{
  "age": 35,
  "income": 50000,
  "savings": 15000
}
```

**Response:**
```json
{
  "approved": 1,
  "approval_probability": 0.87,
  "timestamp": "2026-02-28T10:30:00"
}
```

## 🐳 Docker Deployment

Build and run with Docker:
```bash
docker build -t loan-prediction-api .
docker run -p 8000:8000 loan-prediction-api
```

## 📚 Technologies Used

- **ML Framework**: scikit-learn
- **API**: FastAPI
- **Web UI**: Streamlit
- **Experiment Tracking**: MLflow
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Testing**: pytest

## 🔮 Future Enhancements

- Add more sophisticated models (Random Forest, XGBoost)
- Implement A/B testing framework
- Add model drift detection
- Create CI/CD pipeline
- Add data versioning with DVC
- Implement real-time monitoring dashboard

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.
