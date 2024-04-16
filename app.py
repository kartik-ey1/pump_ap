import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
# Load the models
bearing_model = joblib.load('bearing_model.joblib')
cavitation_model = joblib.load('cavitation_model.pkl')
impeller_model = joblib.load('impeller_model.pkl')
moter_model = joblib.load('moter_model.pkl')
DISPLAY_IMAGE_SIZE = 150



import base64
import streamlit as st
import plotly.express as px

df = px.data.iris()

def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_img_as_base64("image.jpg")
img2 = get_img_as_base64("image2.jpg")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("data:image/jpeg;base64,{img2}");
background-size: 180%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}

[data-testid="stSidebar"] > div:first-child {{
background-image: url("data:image/jpeg;base64,{img}");
background-position: center; 
background-repeat: no-repeat;
background-attachment: fixed;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)










col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.image("download.jpg", caption='IIT ROORKEE', width=DISPLAY_IMAGE_SIZE)

with col2:
    st.markdown("<h3>IIT ROORKEE</h3>", unsafe_allow_html=True)


with col3:
    st.image("download.jpg", caption='IIT ROORKEE', width=DISPLAY_IMAGE_SIZE)


col4, col5, col6 = st.columns([1, 1, 1])

with col4:
    st.write("")

with col5:
    st.image("pump.jpg", caption='PUMP', width=200)

with col6:
     st.write("")



# Function to predict using bearing model
def predict_bearing(df):
    predictions = bearing_model.predict(df[['sensor1', 'sensor3', 'sensor4', 'sensor5']])
    return predictions

# Function to predict using cavitation model
def predict_cavitation(df):
    X = df[['sensor04', 'sensor05', 'sensor06', 'sensor07']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    predictions = cavitation_model.predict(X_scaled)
    return predictions

# Function to predict using impeller model
def predict_impeller(df):
    predictions = impeller_model.predict(df[['sensor09', 'sensor11', 'sensor12']])
    return predictions

# Function to predict using motor model
def predict_motor(df):
    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df[['AccX', 'AccY', 'AccZ']])
    
    # Make predictions
    predictions = moter_model.predict(X_scaled)
    return predictions

# Streamlit web app
def main():
    st.markdown("<h2>Machine Health Conditional Monitoring</h2>", unsafe_allow_html=True)
    st.sidebar.title("Upload CSV File")

    # Upload CSV file
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=['csv'])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Uploaded CSV file:")
        st.write(df)

        # Predictions
        bearing_pred = predict_bearing(df)
        cavitation_pred = predict_cavitation(df)
        impeller_pred = predict_impeller(df)
        motor_pred = predict_motor(df)

        # Display predictions
        st.write("### Predictions:")
        for i in range(len(df)):
            st.markdown(f"## Data point {i+1}:")
            st.write(f"Bearing: {'Bearing_Failure' if bearing_pred[i]==1 else 'No Bearing failure'}")
            st.write(f"Cavitation: {'Cavitation_Failure' if cavitation_pred[i]==1 else 'No Cavitation failure'}")
            st.write(f"Impeller: {'Impeller_Failure' if impeller_pred[i]==1 else 'No Impeller failure'}")
            st.write(f"Motor: {'Moter_Failure' if motor_pred[i]==1 else 'No Moter failure'}")

            # Enhance UI with conditional formatting or charts
            if bearing_pred[i] == 1:
                st.error("Bearing Failure Detected!")
            else:
                st.success("No Bearing Failure")

            if cavitation_pred[i] == 1:
                st.error("Cavitation Failure Detected!")
            else:
                st.success("No Cavitation Failure")

            if impeller_pred[i] == 1:
                st.error("Impeller Failure Detected!")
            else:
                st.success("No Impeller Failure")

            if motor_pred[i] == 1:
                st.error("Motor Failure Detected!")
            else:
                st.success("No Motor Failure")

if __name__ == '__main__':
    main()
