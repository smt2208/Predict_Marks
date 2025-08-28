import streamlit as st
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
import os
import json
from src.config import config

# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .prediction-result {
        font-size: 2rem;
        font-weight: bold;
        color: #2e8b57;
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        background-color: #f0f8f0;
        border: 2px solid #2e8b57;
    }
    .error-message {
        color: #d32f2f;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        background-color: #ffebee;
        border: 2px solid #d32f2f;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1976d2;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main title and description
st.markdown('<h1 class="main-header">📊 Student Performance Predictor</h1>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    # Attempt to load model metadata if present
    metadata_path = os.path.join(config.ARTIFACTS_DIR, "model_metadata.json")
    model_meta = {}
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                model_meta = json.load(f)
        except Exception:
            model_meta = {}
    if model_meta:
        info_extra = f"<br><b>Model:</b> {model_meta.get('best_model_name')} | <b>Val R²:</b> {model_meta.get('best_model_score_val'):.3f} | <b>Test R²:</b> {model_meta.get('test_r2'):.3f}"
    else:
        info_extra = ""
    st.markdown(f'<div class="info-box">This application predicts student math scores based on demographic and academic factors using advanced machine learning algorithms.{info_extra}</div>', unsafe_allow_html=True)

# Check if model artifacts exist (use centralized config paths)
model_path = config.MODEL_FILE_PATH
preprocessor_path = config.PREPROCESSOR_FILE_PATH

if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
    st.error("⚠️ Model artifacts not found! Please train the model first by running: `python src/pipeline/train_pipeline.py`")
    st.stop()

# Sidebar for user input
st.sidebar.header("🔧 Input Features")
st.sidebar.markdown("---")

# Continuous inputs in the sidebar
st.sidebar.subheader("📝 Academic Scores")
writing_score = st.sidebar.slider("Writing Score", min_value=0, max_value=100, value=50, step=1,
                                  help="Student's writing score (0-100)")
reading_score = st.sidebar.slider("Reading Score", min_value=0, max_value=100, value=50, step=1,
                                  help="Student's reading score (0-100)")

# Categorical inputs in the sidebar as dropdowns
st.sidebar.subheader("👤 Demographics")
gender = st.sidebar.selectbox("Gender", ["male", "female"],
                             help="Student's gender")

race_ethnicity = st.sidebar.selectbox("Race/Ethnicity", 
                                     ["group A", "group B", "group C", "group D", "group E"],
                                     help="Student's ethnic group")

st.sidebar.subheader("🎓 Educational Background")
parental_level_of_education = st.sidebar.selectbox(
    "Parental Level of Education", 
    ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"],
    help="Highest education level of student's parents"
)

lunch = st.sidebar.selectbox("Lunch Type", ["standard", "free/reduced"],
                            help="Type of lunch program")

test_preparation_course = st.sidebar.selectbox("Test Preparation Course", ["none", "completed"],
                                              help="Whether student completed test preparation")

# Display current selections
st.sidebar.markdown("---")
st.sidebar.subheader("📋 Current Selections")
st.sidebar.write(f"**Writing Score:** {writing_score}")
st.sidebar.write(f"**Reading Score:** {reading_score}")
st.sidebar.write(f"**Gender:** {gender.title()}")
st.sidebar.write(f"**Race/Ethnicity:** {race_ethnicity.title()}")
st.sidebar.write(f"**Parent Education:** {parental_level_of_education.title()}")
st.sidebar.write(f"**Lunch:** {lunch.title()}")
st.sidebar.write(f"**Test Prep:** {test_preparation_course.title()}")

# Prediction button
st.sidebar.markdown("---")
predict_button = st.sidebar.button("🔮 Predict Math Score", type="primary", use_container_width=True)

# Main content area
if predict_button:
    try:
        # Show loading spinner
        with st.spinner('🔄 Making prediction...'):
            # Create an instance of CustomData using the user inputs
            data = CustomData(
                gender=gender,
                race_ethnicity=race_ethnicity,
                parental_level_of_education=parental_level_of_education,
                lunch=lunch,
                test_preparation_course=test_preparation_course,
                reading_score=reading_score,
                writing_score=writing_score
            )
            
            # Convert data to DataFrame
            pred_df = data.get_data_as_data_frame()
            
            # Create an instance of PredictPipeline and get predictions
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            
            # Validate prediction result
            if results is None or len(results) == 0:
                raise ValueError("Prediction returned empty results")
            
            predicted_score = results[0]
            
            # Ensure the score is within valid range
            predicted_score = max(0, min(100, predicted_score))
            
        # Display the prediction result with nice formatting
        st.markdown("### 🎯 Prediction Result")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f'<div class="prediction-result">Predicted Math Score: {predicted_score:.1f}/100</div>', 
                       unsafe_allow_html=True)
        
        # Performance interpretation
        if predicted_score >= 80:
            performance = "Excellent 🌟"
            color = "#4caf50"
        elif predicted_score >= 70:
            performance = "Good 👍"
            color = "#8bc34a"
        elif predicted_score >= 60:
            performance = "Average 📊"
            color = "#ff9800"
        else:
            performance = "Needs Improvement 📈"
            color = "#f44336"
        
        st.markdown(f"**Performance Level:** <span style='color: {color}; font-weight: bold;'>{performance}</span>", 
                   unsafe_allow_html=True)
        
        # Show feature importance (simplified interpretation)
        st.markdown("### 📊 Key Insights")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Reading Score Impact", f"{reading_score}/100", 
                     help="Higher reading scores typically correlate with better math performance")
            st.metric("Writing Score Impact", f"{writing_score}/100",
                     help="Writing skills often indicate overall academic ability")
        
        with col2:
            prep_impact = "Positive" if test_preparation_course == "completed" else "Limited"
            st.metric("Test Prep Impact", prep_impact,
                     help="Test preparation courses generally improve performance")
            
            lunch_impact = "Standard" if lunch == "standard" else "Free/Reduced"
            st.metric("Socioeconomic Factor", lunch_impact,
                     help="Lunch type can indicate socioeconomic background")
        
    except FileNotFoundError as e:
        st.markdown('<div class="error-message">❌ Model files not found. Please ensure the model has been trained.</div>', 
                   unsafe_allow_html=True)
        st.error(f"Error details: {str(e)}")
        
    except Exception as e:
        st.markdown('<div class="error-message">❌ An error occurred during prediction. Please check your inputs and try again.</div>', 
                   unsafe_allow_html=True)
        st.error(f"Error details: {str(e)}")
        
else:
    # Welcome message and instructions
    st.markdown("### 🚀 Getting Started")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 📝 How to Use:
        1. **Set Academic Scores** - Use the sliders to input reading and writing scores
        2. **Select Demographics** - Choose gender and ethnic group
        3. **Educational Background** - Set parental education level
        4. **Additional Factors** - Select lunch type and test prep status
        5. **Predict** - Click the predict button to get the math score prediction
        """)
    
    with col2:
        st.markdown("""
        #### 🎯 About the Model:
        - **Algorithm**: Uses multiple ML models and selects the best performer
        - **Features**: 7 input features including demographics and academic scores
        - **Accuracy**: Trained on student performance data with cross-validation
        - **Range**: Predicts math scores from 0 to 100
        """)
    
    # Sample data for demonstration
    st.markdown("### 📋 Sample Input Examples")
    
    examples = [
        {"Profile": "High Performer", "Reading": 85, "Writing": 88, "Gender": "Female", "Prep": "Completed"},
        {"Profile": "Average Student", "Reading": 65, "Writing": 62, "Gender": "Male", "Prep": "None"},
        {"Profile": "Improving Student", "Reading": 55, "Writing": 58, "Gender": "Female", "Prep": "Completed"}
    ]
    
    df_examples = pd.DataFrame(examples)
    st.dataframe(df_examples, use_container_width=True)
    
    st.info("💡 **Tip**: Start with one of these examples or use your own values in the sidebar!")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 0.8em;'>"
    "Built with ❤️ using Streamlit and Machine Learning | "
    "Data Science Project</div>", 
    unsafe_allow_html=True
)
