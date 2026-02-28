"""
FastAPI REST API for Loan Prediction System

This API provides endpoints for:
- Health checks
- Single predictions
- Batch predictions
- Model information
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any
import uvicorn
from datetime import datetime
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from src.models.predict import LoanPredictor
from src.utils.logger import setup_logger

app = FastAPI(
    title="Loan Prediction API",
    description="REST API for predicting loan approval using machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = setup_logger(__name__, "api")

try:
    predictor = LoanPredictor()
    logger.info("LoanPredictor initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize predictor: {str(e)}")
    predictor = None


class LoanApplication(BaseModel):
    """Schema for single loan application."""
    age: int = Field(..., ge=18, le=100, description="Applicant's age (18-100)")
    income: float = Field(..., ge=0, description="Annual income in dollars")
    savings: float = Field(..., ge=0, description="Savings amount in dollars")
    
    @validator('income')
    def validate_income(cls, v):
        if v > 10000000:
            raise ValueError('Income seems unrealistically high')
        return v
    
    @validator('savings')
    def validate_savings(cls, v):
        if v > 10000000:
            raise ValueError('Savings amount seems unrealistically high')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "age": 35,
                "income": 50000,
                "savings": 15000
            }
        }


class BatchLoanApplication(BaseModel):
    """Schema for batch loan applications."""
    applications: List[LoanApplication] = Field(..., min_items=1, max_items=100)
    
    class Config:
        schema_extra = {
            "example": {
                "applications": [
                    {"age": 35, "income": 50000, "savings": 15000},
                    {"age": 28, "income": 40000, "savings": 8000}
                ]
            }
        }


class PredictionResponse(BaseModel):
    """Schema for prediction response."""
    approved: int
    approval_status: str
    probability: float
    confidence: float
    input_data: Dict[str, Any]
    timestamp: str


class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction response."""
    total_applications: int
    approved_count: int
    rejected_count: int
    predictions: List[Dict[str, Any]]
    timestamp: str


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - API status."""
    return {
        "message": "Loan Prediction API is running",
        "version": "1.0.0",
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictor service is not available"
        )
    
    return {
        "status": "healthy",
        "model_loaded": predictor.model is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/model/info", tags=["Model"])
async def model_info():
    """Get information about the loaded model."""
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictor service is not available"
        )
    
    try:
        info = {
            "model_type": type(predictor.model).__name__,
            "features": predictor.config['features']['input_features'],
            "model_path": predictor.config['model_path'],
            "timestamp": datetime.now().isoformat()
        }
        
        if hasattr(predictor.model, 'coef_'):
            info['coefficients'] = predictor.model.coef_.tolist()
        
        return info
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving model information: {str(e)}"
        )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_loan(application: LoanApplication):
    """
    Predict loan approval for a single application.
    
    Returns:
    - approved: 1 for approved, 0 for rejected
    - approval_status: "Approved" or "Rejected"
    - probability: Probability of approval (0-1)
    - confidence: Model confidence in prediction
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictor service is not available"
        )
    
    try:
        logger.info(f"Received prediction request: {application.dict()}")
        
        result = predictor.predict(
            age=application.age,
            income=application.income,
            savings=application.savings
        )
        
        logger.info(f"Prediction result: {result['approval_status']}")
        
        return result
    
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error making prediction: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(batch: BatchLoanApplication):
    """
    Predict loan approval for multiple applications.
    
    Maximum 100 applications per request.
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictor service is not available"
        )
    
    try:
        logger.info(f"Received batch prediction request with {len(batch.applications)} applications")
        
        applications_data = [app.dict() for app in batch.applications]
        
        results = predictor.predict_batch(applications_data)
        
        approved_count = sum(1 for r in results if r['approved'] == 1)
        rejected_count = len(results) - approved_count
        
        logger.info(f"Batch predictions completed: {approved_count} approved, {rejected_count} rejected")
        
        return {
            "total_applications": len(results),
            "approved_count": approved_count,
            "rejected_count": rejected_count,
            "predictions": results,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error making batch predictions: {str(e)}"
        )


@app.post("/predict/explain", tags=["Prediction"])
async def predict_with_explanation(application: LoanApplication):
    """
    Predict loan approval with explanation of factors.
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictor service is not available"
        )
    
    try:
        logger.info(f"Received prediction with explanation request: {application.dict()}")
        
        explanation = predictor.explain_prediction(
            age=application.age,
            income=application.income,
            savings=application.savings
        )
        
        return explanation
    
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Explanation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating explanation: {str(e)}"
        )


if __name__ == "__main__":
    logger.info("Starting Loan Prediction API server...")
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
