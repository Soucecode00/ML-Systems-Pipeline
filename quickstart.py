"""
Quick start script to train the model and run a sample prediction
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

print("="*70)
print("LOAN PREDICTION SYSTEM - QUICK START")
print("="*70)

print("\nStep 1: Training the model...")
print("-" * 70)

from src.models.train_model import ModelTrainer

trainer = ModelTrainer()
results, X_train, X_test, y_train, y_test = trainer.train_pipeline(perform_tuning=False)

print(f"\n✓ Model trained successfully!")
print(f"  - Training samples: {results['training_samples']}")
print(f"  - Test samples: {results['test_samples']}")
print(f"  - CV Score: {results['cv_results']['mean_cv_score']:.4f}")

print("\nStep 2: Evaluating the model...")
print("-" * 70)

from src.models.evaluate_model import ModelEvaluator

evaluator = ModelEvaluator()
evaluation_results = evaluator.generate_evaluation_report(trainer.model, X_test, y_test)

print(f"\n✓ Model evaluated successfully!")
print(f"  - Accuracy: {evaluation_results['metrics']['accuracy']:.4f}")
print(f"  - Precision: {evaluation_results['metrics']['precision']:.4f}")
print(f"  - Recall: {evaluation_results['metrics']['recall']:.4f}")
print(f"  - F1-Score: {evaluation_results['metrics']['f1_score']:.4f}")

print("\nStep 3: Making sample predictions...")
print("-" * 70)

from src.models.predict import LoanPredictor

predictor = LoanPredictor()

test_cases = [
    {"age": 35, "income": 50000, "savings": 15000},
    {"age": 25, "income": 30000, "savings": 5000},
    {"age": 45, "income": 80000, "savings": 30000}
]

for i, case in enumerate(test_cases, 1):
    result = predictor.predict(**case)
    print(f"\nTest Case {i}:")
    print(f"  Input: Age={case['age']}, Income=${case['income']:,}, Savings=${case['savings']:,}")
    print(f"  Result: {result['approval_status']}")
    print(f"  Probability: {result['probability']:.2%}")

print("\n" + "="*70)
print("SETUP COMPLETE!")
print("="*70)
print("\nNext steps:")
print("  1. Start the API server: python api.py")
print("  2. Start the web interface: streamlit run streamlit_app.py")
print("  3. Access API docs: http://localhost:8000/docs")
print("  4. Access web UI: http://localhost:8501")
print("\nFor Docker deployment, see DOCKER.md")
print("="*70)
