import streamlit as st
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Set up the main title and description
st.title("Student Performance Predictor")
st.write("This app predicts the math score based on other parameters.")

# Sidebar for user input
st.sidebar.header("Input Features")

# Continuous inputs in the sidebar
writing_score = st.sidebar.number_input("Writing Score", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
reading_score = st.sidebar.number_input("Reading Score", min_value=0.0, max_value=100.0, value=50.0, step=1.0)

# Categorical inputs in the sidebar as dropdowns
gender = st.sidebar.selectbox("Gender", ["male", "female"])
race_ethnicity = st.sidebar.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
parental_level_of_education = st.sidebar.selectbox(
    "Parental Level of Education", 
    ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"]
)
lunch = st.sidebar.selectbox("Lunch", ["standard", "free/reduced"])
test_preparation_course = st.sidebar.selectbox("Test Preparation Course", ["none", "completed"])

# When the 'Predict' button is clicked
if st.sidebar.button("Predict"):
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
    try:
        results = predict_pipeline.predict(pred_df)
        # Display the formatted prediction result
        st.write("### Prediction Result:")
        st.write(f"On the basis of the inputs, the predicted math score is {results[0]:.2f}")
    except Exception as e:
        st.write("Error during prediction:", e)
else:
    st.write("Use the sidebar to input features and click 'Predict'.")
