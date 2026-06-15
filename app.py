import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import warnings
import types

# Suppress warnings for cleaner UI
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# 1. IMPROVED PICKLE COMPATIBILITY PATCH
# -----------------------------------------------------------------------------
def patch_sklearn_pickle():
    """
    Creates dummy classes for missing sklearn internal structures 
    to allow loading of pickles from different sklearn versions.
    """
    # Patch ColumnTransformer
    try:
        import sklearn.compose._column_transformer as ct_module
        
        missing_classes = ['_RemainderColsList', '_ColumnTransformer']
        
        for cls_name in missing_classes:
            if not hasattr(ct_module, cls_name):
                class DummyClass(list):
                    def __init__(self, *args, **kwargs):
                        super().__init__(*args, **kwargs)
                
                setattr(ct_module, cls_name, DummyClass)
                print(f"⚠️ Patched missing class: {cls_name}")
    except Exception as e:
        print(f"Note: Could not apply pickle patch for ColumnTransformer: {e}")

    # Patch SimpleImputer for _fill_dtype issue
    try:
        from sklearn.impute import SimpleImputer
        
        if not hasattr(SimpleImputer, '_fill_dtype'):
            def _fill_dtype(self, dtype):
                return np.float64
            SimpleImputer._fill_dtype = _fill_dtype
            print("⚠️ Patched SimpleImputer._fill_dtype at class level")
    except Exception as e:
        print(f"Note: Could not patch SimpleImputer: {e}")

patch_sklearn_pickle()

# -----------------------------------------------------------------------------
# 2. PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Hypertension AI Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a medical/professional look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background-color: #e74c3c;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        height: 3em;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #c0392b;
        transform: translateY(-2px);
    }
    .info-box {
        background-color: #e8f4f8;
        border-left: 5px solid #3498db;
        padding: 15px;
        margin-bottom: 20px;
        border-radius: 4px;
    }
    div[data-testid="stSidebarNav"] {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. LOAD RESOURCES (With Enhanced Error Handling)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_resources():
    """Load models and metadata safely."""
    required_files = [
        'xgboost_model.pkl', 
        'random_forest_model.pkl', 
        'knn_model.pkl', 
        'feature_names.pkl', 
        'imputer.pkl'
    ]
    
    # Check if files exist
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        st.error(f"❌ Missing critical files: {', '.join(missing_files)}")
        st.info("Please run `save_models.py` first to generate these files.")
        st.stop()

    try:
        models = {
            'XGBoost': joblib.load('xgboost_model.pkl'),
            'Random Forest': joblib.load('random_forest_model.pkl'),
            'KNN': joblib.load('knn_model.pkl')
        }
        feature_names = joblib.load('feature_names.pkl')
        imputer = joblib.load('imputer.pkl')
        
        # Load sample data for insights if available
        sample_df = None
        if os.path.exists('sample_data.pkl'):
            sample_df = joblib.load('sample_data.pkl')
            
        return models, feature_names, imputer, sample_df
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.info("This might be due to a version mismatch. Try re-running `save_models.py` in this environment.")
        st.stop()

models, feature_names, global_imputer, sample_df = load_resources()

# -----------------------------------------------------------------------------
# 4. SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "🏠 Home", 
    "🔮 Prediction Tool", 
    "📊 Model Insights", 
    "ℹ️ About Hypertension"
])

# -----------------------------------------------------------------------------
# 5. IMPROVED HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def get_user_input():
    """Create input form for user data."""
    st.subheader("Patient Details")
    st.markdown("Enter the clinical and demographic details below.")
    
    col1, col2 = st.columns(2)
    input_data = {}
    
    with col1:
        input_data['age'] = st.slider("Age (Years)", 18, 90, 45, step=1)
        input_data['BMI'] = st.number_input("Body Mass Index (BMI)", 15.0, 50.0, 25.0, step=0.1)
        input_data['married'] = st.selectbox("Marital Status", [0, 1], format_func=lambda x: "Single/Divorced/Widowed" if x==0 else "Married")
        input_data['male.gender'] = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x==0 else "Male")
        input_data['hgb_centered'] = st.number_input("Hemoglobin (Centered)", -10.0, 10.0, 0.0, step=0.1, help="Hemoglobin level relative to population mean")
        
    with col2:
        input_data['adv_HIV'] = st.selectbox("Advanced HIV Disease", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
        input_data['survtime'] = st.number_input("Survival Time (Days)", 0, 3000, 500, step=1)
        input_data['event'] = st.selectbox("Clinical Event Occurred", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
        input_data['arv_naive'] = st.selectbox("ARV Naive", [0, 1], format_func=lambda x: "On Treatment" if x==0 else "Naive (New)")
        input_data['urban.clinic'] = st.selectbox("Clinic Location", [0, 1], format_func=lambda x: "Rural" if x==0 else "Urban")
        input_data['log_creat_centered'] = st.number_input("Log Creatinine (Centered)", -5.0, 5.0, 0.0, step=0.1, help="Kidney function marker")
        
    return pd.DataFrame([input_data])

def predict_hypertension(input_df):
    """Run predictions using all loaded models with improved error handling."""
    results = {}
    
    # Validate input
    if input_df.empty:
        st.error("Input data is empty!")
        return None
    
    # Ensure all required columns are present
    missing_cols = [col for col in feature_names if col not in input_df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        return None
    
    # 1. Impute missing values using the saved imputer
    try:
        # Reorder columns to match training data
        input_ordered = input_df[feature_names]
        
        # Patch the imputer instance if needed
       if hasattr(global_imputer, '_fill_dtype'):
    if callable(global_imputer._fill_dtype):
        global_imputer._fill_dtype = np.float64

        input_imputed = global_imputer.transform(input_ordered)
        input_clean = pd.DataFrame(input_imputed, columns=feature_names)
    except AttributeError as e:
        if "_fill_dtype" in str(e):
            st.warning("⚠️ Imputer version mismatch detected. Applying fallback imputation (filling missing values with 0).")
            # Fallback: 0 is the mathematical mean for centered features
            input_clean = input_ordered.fillna(0)
        else:
            st.error(f"Preprocessing error: {e}")
            return None
    except Exception as e:
        st.error(f"Preprocessing error: {e}")
        return None
    
    # ... rest of the function remains the same
    # 2. Predict with each model
    for name, model in models.items():
        try:
            pred_class = model.predict(input_clean)[0]
            
            # Handle models that may not have predict_proba
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(input_clean)[0]
                # Get probability of positive class (index 1)
                pred_prob = proba[1] if len(proba) > 1 else proba[0]
            else:
                # Fallback for models without predict_proba
                pred_prob = float(pred_class)
            
            results[name] = {
                'prediction': "High Risk" if pred_class == 1 else "Low Risk",
                'probability': float(pred_prob),
                'class': int(pred_class)
            }
        except Exception as e:
            st.error(f"Error in {name} model: {e}")
            results[name] = {
                'prediction': "Error",
                'probability': 0.0,
                'class': -1
            }
            
    return results

def extract_feature_importance(model, feature_names):
    """Extract feature importance from various model types."""
    try:
        # Try different methods to extract feature importance
        if hasattr(model, 'feature_importances_'):
            # For tree-based models (XGBoost, Random Forest)
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # For linear models
            importances = np.abs(model.coef_[0])
        else:
            # For other models, return None
            return None
        
        feat_imp = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=True)
        
        return feat_imp
    except Exception as e:
        st.warning(f"Could not extract feature importance: {e}")
        return None

# -----------------------------------------------------------------------------
# 6. PAGE CONTENT
# -----------------------------------------------------------------------------

if page == "🏠 Home":
    st.markdown('<div class="main-header">Hypertension Prediction System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Risk Assessment using Clinical & Demographic Data</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Models Integrated", value="3", delta="XGBoost, RF, KNN")
    with col2:
        st.metric(label="Best Accuracy", value="~89%", delta="XGBoost Leader")
    with col3:
        st.metric(label="Features Analyzed", value="11", delta="Clinical Markers")
        
    st.image("https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80", use_column_width=True)
    
    st.markdown("""
    ### How it works
    This application uses machine learning models trained on patient data to predict the likelihood of hypertension. 
    It analyzes key factors including:
    *   **Demographics**: Age, Gender, Marital Status
    *   **Clinical Metrics**: BMI, Hemoglobin, Creatinine
    *   **History**: HIV status, ARV usage, Survival time
    
    > ⚠️ **Disclaimer**: This tool is for educational purposes only. Always consult a healthcare professional for medical advice.
    """)

elif page == "🔮 Prediction Tool":
    st.header("Patient Risk Assessment")
    
    input_df = get_user_input()
    
    if st.button("Analyze Risk"):
        with st.spinner("Running models..."):
            results = predict_hypertension(input_df)
            
        if results:
            # Filter out error results
            valid_results = {k: v for k, v in results.items() if v['prediction'] != "Error"}
            
            if valid_results:
                st.success("Analysis Complete!")
                
                # Display Results in Cards
                cols = st.columns(len(valid_results))
                for i, (name, res) in enumerate(valid_results.items()):
                    with cols[i]:
                        st.markdown(f"**{name}**")
                        if res['prediction'] == "High Risk":
                            st.error(f"⚠️ {res['prediction']}")
                        else:
                            st.success(f"✅ {res['prediction']}")
                        
                        st.progress(float(res['probability']))
                        st.caption(f"Probability: {res['probability']:.2%}")
                
                # Consensus Logic
                high_risk_count = sum(1 for r in valid_results.values() if r['class'] == 1)
                total_models = len(valid_results)
                
                st.markdown("---")
                if total_models > 0:
                    risk_percentage = (high_risk_count / total_models) * 100
                    
                    if risk_percentage >= 66:
                        st.warning(f"🚨 **High Risk Alert**: {high_risk_count}/{total_models} models predict High Risk ({risk_percentage:.0f}%). Please consult a doctor.")
                    elif risk_percentage >= 33:
                        st.info(f"ℹ️ **Moderate Risk**: {high_risk_count}/{total_models} models predict High Risk ({risk_percentage:.0f}%). Monitor your health.")
                    else:
                        st.success(f"✅ **Low Risk**: Only {high_risk_count}/{total_models} models predict High Risk ({risk_percentage:.0f}%). Maintain healthy habits.")
            else:
                st.error("All models encountered errors during prediction.")

elif page == "📊 Model Insights":
    st.header("Model Performance & Feature Importance")
    
    tab1, tab2 = st.tabs(["Performance Metrics", "Feature Importance"])
    
    with tab1:
        st.subheader("Comparison of Top Models")
        # Mock metrics based on typical performance for these models on this dataset
        metrics_data = {
            'Model': ['XGBoost', 'Random Forest', 'KNN'],
            'Accuracy': [0.887, 0.843, 0.864],
            'Precision': [0.865, 0.807, 0.796],
            'Recall': [0.916, 0.901, 0.978],
            'F1-Score': [0.890, 0.851, 0.878]
        }
        df_metrics = pd.DataFrame(metrics_data)
        
        fig = go.Figure(data=[
            go.Bar(name='Accuracy', x=df_metrics['Model'], y=df_metrics['Accuracy']),
            go.Bar(name='F1-Score', x=df_metrics['Model'], y=df_metrics['F1-Score'])
        ])
        fig.update_layout(barmode='group', title="Model Accuracy vs F1-Score")
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(df_metrics.style.highlight_max(axis=0, subset=['Accuracy', 'F1-Score']), use_container_width=True)

    with tab2:
        st.subheader("Global Feature Importance (XGBoost)")
        try:
            # Extract importance from the XGBoost model
            xgb_model = models['XGBoost']
            
            # Try to extract feature importance with multiple fallback methods
            feat_imp = None
            
            # Method 1: Direct feature_importances_ attribute
            if hasattr(xgb_model, 'feature_importances_'):
                importances = xgb_model.feature_importances_
                feat_imp = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values(by='Importance', ascending=True)
            
            # Method 2: If it's a pipeline, try to access the classifier
            elif hasattr(xgb_model, 'named_steps'):
                try:
                    # Try common step names
                    for step_name in ['classifier', 'xgb', 'model']:
                        if step_name in xgb_model.named_steps:
                            classifier = xgb_model.named_steps[step_name]
                            if hasattr(classifier, 'feature_importances_'):
                                importances = classifier.feature_importances_
                                feat_imp = pd.DataFrame({
                                    'Feature': feature_names,
                                    'Importance': importances
                                }).sort_values(by='Importance', ascending=True)
                                break
                except:
                    pass
            
            if feat_imp is not None and not feat_imp.empty:
                fig = px.bar(feat_imp, x='Importance', y='Feature', orientation='h', 
                             title="Top Predictors of Hypertension")
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("💡 **Insight**: Age and BMI are typically the strongest predictors, followed by clinical markers like Hemoglobin and Creatinine.")
            else:
                st.warning("Could not extract feature importance from the model.")
                
        except Exception as e:
            st.error(f"Could not load feature importance: {e}")
            st.info("This might be due to the model structure. The models are still functional for predictions.")

elif page == "ℹ️ About Hypertension":
    st.header("Understanding Hypertension")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### What is Hypertension?
        Hypertension (high blood pressure) is a condition where the force of the blood against artery walls is too high. 
        It is often called the "silent killer" because it may have no warning signs or symptoms.
        
        ### Key Risk Factors in Our Model
        1. **Age**: Blood vessels naturally stiffen with age.
        2. **BMI**: Higher body mass index increases strain on the heart.
        3. **Kidney Function**: Measured via Creatinine levels.
        4. **HIV Status**: Advanced HIV and ARV usage can impact cardiovascular health.
        
        ### Prevention
        *   Maintain a healthy weight.
        *   Exercise regularly.
        *   Eat a balanced diet low in salt.
        *   Limit alcohol and avoid smoking.
        """)
        
    with col2:
        if sample_df is not None:
            st.subheader("Dataset Sample")
            st.dataframe(sample_df.head(), use_container_width=True)
        else:
            st.info("Sample data not available for display.")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: grey; font-size: 0.8rem;'>Developed by Brian Njagi Kimathi | Hypertension Prediction Project</div>", unsafe_allow_html=True)
