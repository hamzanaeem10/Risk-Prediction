"""
Financial Risk Prediction - Streamlit Frontend
A modern, sleek UI for predicting loan risk
"""
import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
import os

# Page Configuration
st.set_page_config(
    page_title="Financial Risk Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    .css-1d391kg {
        background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
    }
    h1 {
        background: linear-gradient(90deg, #00d9ff, #00ff88);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem !important;
    }
    .stButton>button {
        background: linear-gradient(90deg, #00d9ff, #00ff88);
        color: #1a1a2e;
        font-weight: bold;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-size: 1.1rem;
        transition: transform 0.3s, box-shadow 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(0, 217, 255, 0.3);
    }
    .risk-card {
        background: rgba(255,255,255,0.05);
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
    }
    .metric-value {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
    }
    .low-risk { color: #00ff88; }
    .medium-risk { color: #ffa500; }
    .high-risk { color: #ff4757; }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_URL = os.environ.get("API_URL", "http://localhost:8000")

def check_api_health():
    """Check if API is running."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def predict_risk(data):
    """Call prediction API."""
    try:
        response = requests.post(f"{API_URL}/predict", json=data, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

def create_gauge_chart(probability, risk_class):
    """Create a gauge chart for risk visualization."""
    colors = {"Low": "#00ff88", "Medium": "#ffa500", "High": "#ff4757"}
    color = colors.get(risk_class, "#00d9ff")
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Risk Level: {risk_class}", 'font': {'size': 24, 'color': 'white'}},
        number={'suffix': "%", 'font': {'size': 40, 'color': color}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': 'white'},
            'bar': {'color': color},
            'bgcolor': 'rgba(255,255,255,0.1)',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 33], 'color': 'rgba(0,255,136,0.2)'},
                {'range': [33, 66], 'color': 'rgba(255,165,0,0.2)'},
                {'range': [66, 100], 'color': 'rgba(255,71,87,0.2)'}
            ],
            'threshold': {
                'line': {'color': color, 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        height=300
    )
    return fig

def create_probability_chart(probabilities):
    """Create bar chart for class probabilities."""
    classes = list(probabilities.keys())
    values = [v * 100 for v in probabilities.values()]
    colors = ['#00ff88', '#ffa500', '#ff4757']
    
    fig = go.Figure(go.Bar(
        x=classes,
        y=values,
        marker_color=colors,
        text=[f'{v:.1f}%' for v in values],
        textposition='auto'
    ))
    
    fig.update_layout(
        title='Probability Distribution',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        height=250,
        yaxis={'title': 'Probability (%)', 'range': [0, 100]},
        xaxis={'title': ''}
    )
    return fig

# ========== MAIN APP ==========

st.title("💰 Financial Risk Predictor")
st.markdown("### AI-Powered Loan Risk Assessment")

# API Status
api_status = check_api_health()
if api_status:
    st.sidebar.success("✅ API Online")
else:
    st.sidebar.error("❌ API Offline - Start with: `uvicorn src.api.main:app`")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Instructions")
st.sidebar.markdown("""
1. Fill in applicant details
2. Click **Predict Risk**
3. View AI assessment
""")

# ========== INPUT FORM ==========

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 👤 Applicant Info")
    age = st.slider("Age", 18, 80, 35)
    gender = st.selectbox("Gender", ["Male", "Female"])
    education = st.selectbox("Education", ["High School", "Bachelor's", "Master's", "PhD"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
    employment = st.selectbox("Employment", ["Employed", "Self-Employed", "Unemployed"])
    years_at_job = st.slider("Years at Job", 0, 40, 5)

with col2:
    st.markdown("#### 💵 Financial Info")
    income = st.number_input("Annual Income ($)", 10000, 500000, 75000, step=5000)
    credit_score = st.slider("Credit Score", 300, 850, 700)
    loan_amount = st.number_input("Loan Amount ($)", 1000, 500000, 25000, step=1000)
    loan_purpose = st.selectbox("Loan Purpose", ["Auto", "Home", "Education", "Business", "Personal", "Debt Consolidation"])
    assets_value = st.number_input("Total Assets ($)", 0, 2000000, 150000, step=10000)

with col3:
    st.markdown("#### 📊 Risk Factors")
    debt_to_income = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.25, step=0.05)
    payment_history = st.selectbox("Payment History", ["Good", "Fair", "Poor"])
    previous_defaults = st.slider("Previous Defaults", 0, 10, 0)
    dependents = st.slider("Number of Dependents", 0, 10, 2)
    marital_change = st.slider("Marital Status Changes", 0, 5, 0)

st.markdown("---")

# ========== PREDICTION ==========

if st.button("🔮 Predict Risk", use_container_width=True):
    if not api_status:
        st.error("API is offline. Please start the API first.")
    else:
        with st.spinner("Analyzing risk factors..."):
            # Prepare data
            data = {
                "age": age,
                "gender": gender,
                "education_level": education,
                "marital_status": marital_status,
                "income": float(income),
                "credit_score": float(credit_score),
                "loan_amount": float(loan_amount),
                "loan_purpose": loan_purpose,
                "employment_status": employment,
                "years_at_current_job": float(years_at_job),
                "payment_history": payment_history,
                "debt_to_income_ratio": float(debt_to_income),
                "assets_value": float(assets_value),
                "number_of_dependents": int(dependents),
                "previous_defaults": float(previous_defaults),
                "marital_status_change": int(marital_change)
            }
            
            result = predict_risk(data)
            
            if "error" in result:
                st.error(f"Prediction failed: {result['error']}")
            else:
                # Display Results
                st.markdown("---")
                st.markdown("## 📊 Risk Assessment Results")
                
                col_result1, col_result2 = st.columns(2)
                
                with col_result1:
                    risk_class = result['risk_class']
                    confidence = result['confidence']
                    
                    # Gauge Chart
                    fig_gauge = create_gauge_chart(confidence, risk_class)
                    st.plotly_chart(fig_gauge, use_container_width=True)
                
                with col_result2:
                    # Probability Distribution
                    fig_prob = create_probability_chart(result['risk_probabilities'])
                    st.plotly_chart(fig_prob, use_container_width=True)
                
                # Recommendation
                st.markdown("---")
                recommendations = {
                    "Low": ("✅ **Low Risk** - Applicant shows strong financial health. Recommended for approval.", "success"),
                    "Medium": ("⚠️ **Medium Risk** - Additional verification recommended. Consider adjusted terms.", "warning"),
                    "High": ("🚨 **High Risk** - Significant risk factors present. Requires thorough review.", "error")
                }
                
                rec_text, rec_type = recommendations.get(risk_class, ("Unknown", "info"))
                getattr(st, rec_type)(rec_text)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>Built with Streamlit • XGBoost • FastAPI</p>",
    unsafe_allow_html=True
)
