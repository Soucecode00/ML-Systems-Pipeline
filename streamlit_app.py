"""
Streamlit Web Interface for Loan Prediction System

This provides a user-friendly web interface for:
- Single loan predictions
- Batch predictions via file upload
- Model performance visualization
- Prediction history
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from src.models.predict import LoanPredictor


st.set_page_config(
    page_title="Loan Prediction System",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .approved {
        color: #28a745;
        font-weight: bold;
    }
    .rejected {
        color: #dc3545;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    """Load the predictor model (cached)."""
    try:
        return LoanPredictor()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please train the model first by running: python -m src.models.train_model")
        return None


def main():
    """Main application."""
    
    st.markdown('<h1 class="main-header">💰 Loan Prediction System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    predictor = load_predictor()
    
    if predictor is None:
        st.stop()
    
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose a page:",
        ["🏠 Home", "🔮 Single Prediction", "📊 Batch Prediction", "📈 Model Info", "ℹ️ About"]
    )
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "🔮 Single Prediction":
        show_single_prediction_page(predictor)
    elif page == "📊 Batch Prediction":
        show_batch_prediction_page(predictor)
    elif page == "📈 Model Info":
        show_model_info_page(predictor)
    elif page == "ℹ️ About":
        show_about_page()


def show_home_page():
    """Display home page."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🎯 Features
        - Real-time loan predictions
        - Batch processing
        - Model explanations
        - Performance metrics
        """)
    
    with col2:
        st.markdown("""
        ### 🚀 Quick Start
        1. Navigate to Single Prediction
        2. Enter applicant details
        3. Get instant approval decision
        4. View probability score
        """)
    
    with col3:
        st.markdown("""
        ### 📊 Capabilities
        - Machine Learning powered
        - High accuracy predictions
        - Transparent decision making
        - Easy to use interface
        """)
    
    st.markdown("---")
    st.info("👈 Use the sidebar to navigate between different sections")


def show_single_prediction_page(predictor):
    """Display single prediction page."""
    st.markdown('<h2 class="sub-header">🔮 Single Loan Prediction</h2>', unsafe_allow_html=True)
    
    st.write("Enter the applicant's information below to get a loan approval prediction.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", min_value=18, max_value=100, value=35, help="Applicant's age")
        income = st.number_input(
            "Annual Income ($)", 
            min_value=0, 
            max_value=1000000, 
            value=50000, 
            step=1000,
            help="Annual income in dollars"
        )
    
    with col2:
        savings = st.number_input(
            "Savings ($)", 
            min_value=0, 
            max_value=1000000, 
            value=15000, 
            step=1000,
            help="Total savings amount"
        )
        
        savings_ratio = savings / (income + 1) if income > 0 else 0
        st.metric("Savings to Income Ratio", f"{savings_ratio:.2%}")
    
    st.markdown("---")
    
    if st.button("🔍 Predict Loan Approval", type="primary", use_container_width=True):
        with st.spinner("Making prediction..."):
            try:
                result = predictor.predict(age=age, income=income, savings=savings)
                
                st.success("Prediction completed!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    approval_class = "approved" if result['approved'] == 1 else "rejected"
                    st.markdown(
                        f'<div class="metric-card"><h3>Decision</h3>'
                        f'<p class="{approval_class}">{result["approval_status"]}</p></div>',
                        unsafe_allow_html=True
                    )
                
                with col2:
                    st.markdown(
                        f'<div class="metric-card"><h3>Approval Probability</h3>'
                        f'<p style="font-size: 1.5rem; font-weight: bold;">{result["probability"]:.2%}</p></div>',
                        unsafe_allow_html=True
                    )
                
                with col3:
                    st.markdown(
                        f'<div class="metric-card"><h3>Confidence</h3>'
                        f'<p style="font-size: 1.5rem; font-weight: bold;">{result["confidence"]:.2%}</p></div>',
                        unsafe_allow_html=True
                    )
                
                st.markdown("---")
                
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=result['probability'] * 100,
                    title={'text': "Approval Probability"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#28a745" if result['approved'] == 1 else "#dc3545"},
                        'steps': [
                            {'range': [0, 33], 'color': "#ffcccc"},
                            {'range': [33, 67], 'color': "#fff9cc"},
                            {'range': [67, 100], 'color': "#ccffcc"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                
                st.plotly_chart(fig, use_container_width=True)
                
                explanation = predictor.explain_prediction(age=age, income=income, savings=savings)
                if explanation['factors']:
                    st.markdown("### 📋 Key Factors")
                    for factor in explanation['factors']:
                        st.info(factor)
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")


def show_batch_prediction_page(predictor):
    """Display batch prediction page."""
    st.markdown('<h2 class="sub-header">📊 Batch Loan Prediction</h2>', unsafe_allow_html=True)
    
    st.write("Upload a CSV file with multiple loan applications to get predictions.")
    
    st.markdown("""
    **Required columns:**
    - `age`: Applicant's age (18-100)
    - `income`: Annual income in dollars
    - `savings`: Savings amount in dollars
    """)
    
    sample_data = pd.DataFrame({
        'age': [35, 28, 45, 22, 55],
        'income': [50000, 40000, 70000, 25000, 80000],
        'savings': [15000, 8000, 25000, 2000, 30000]
    })
    
    with st.expander("📄 View Sample Data Format"):
        st.dataframe(sample_data)
        csv = sample_data.to_csv(index=False)
        st.download_button(
            label="Download Sample CSV",
            data=csv,
            file_name="sample_loan_applications.csv",
            mime="text/csv"
        )
    
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            st.write(f"Loaded {len(df)} applications")
            st.dataframe(df.head())
            
            if st.button("🚀 Run Batch Predictions", type="primary"):
                with st.spinner(f"Processing {len(df)} applications..."):
                    results = predictor.predict_batch(df)
                    
                    results_df = pd.DataFrame(results)
                    df_with_results = df.copy()
                    df_with_results['approved'] = results_df['approved']
                    df_with_results['approval_status'] = results_df['approval_status']
                    df_with_results['probability'] = results_df['probability']
                    
                    st.success("Batch predictions completed!")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Applications", len(results))
                    with col2:
                        approved = sum(r['approved'] for r in results)
                        st.metric("Approved", approved, delta=f"{approved/len(results)*100:.1f}%")
                    with col3:
                        rejected = len(results) - approved
                        st.metric("Rejected", rejected, delta=f"{rejected/len(results)*100:.1f}%")
                    
                    st.markdown("---")
                    
                    fig = px.pie(
                        values=[approved, rejected],
                        names=['Approved', 'Rejected'],
                        title='Approval Distribution',
                        color_discrete_sequence=['#28a745', '#dc3545']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("### 📊 Detailed Results")
                    st.dataframe(df_with_results)
                    
                    csv_results = df_with_results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv_results,
                        file_name=f"loan_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")


def show_model_info_page(predictor):
    """Display model information page."""
    st.markdown('<h2 class="sub-header">📈 Model Information</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Model Details")
        st.info(f"**Model Type:** {type(predictor.model).__name__}")
        st.info(f"**Features:** {', '.join(predictor.config['features']['input_features'])}")
    
    with col2:
        st.markdown("### Model Configuration")
        st.json(predictor.config['model'])
    
    if hasattr(predictor.model, 'coef_'):
        st.markdown("### Feature Coefficients")
        
        features = predictor.config['features']['input_features']
        coef = predictor.model.coef_[0][:len(features)]
        
        coef_df = pd.DataFrame({
            'Feature': features,
            'Coefficient': coef,
            'Abs_Coefficient': np.abs(coef)
        }).sort_values('Abs_Coefficient', ascending=False)
        
        fig = px.bar(
            coef_df,
            x='Coefficient',
            y='Feature',
            orientation='h',
            title='Feature Coefficients',
            color='Coefficient',
            color_continuous_scale='RdBu_r'
        )
        st.plotly_chart(fig, use_container_width=True)


def show_about_page():
    """Display about page."""
    st.markdown('<h2 class="sub-header">ℹ️ About</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Loan Prediction System
    
    This is a complete end-to-end machine learning system for predicting loan approvals.
    
    **Key Components:**
    - **Data Pipeline**: Automated data ingestion and preprocessing
    - **Feature Engineering**: Advanced feature creation and scaling
    - **Model Training**: Logistic Regression with hyperparameter tuning
    - **Model Evaluation**: Comprehensive metrics and visualizations
    - **REST API**: FastAPI-based API for predictions
    - **Web Interface**: Interactive Streamlit dashboard
    
    **Technology Stack:**
    - Python 3.x
    - scikit-learn
    - FastAPI
    - Streamlit
    - Pandas & NumPy
    - Plotly
    
    **Model Performance:**
    The model is trained on historical loan data and uses features like age, income, 
    and savings to predict loan approval probability.
    
    **How it works:**
    1. Data is preprocessed and features are engineered
    2. Model makes predictions based on learned patterns
    3. Results include approval decision and probability
    4. Explanations help understand the decision factors
    
    ---
    
    **Version:** 1.0.0  
    **Last Updated:** February 2026
    """)


if __name__ == "__main__":
    main()
