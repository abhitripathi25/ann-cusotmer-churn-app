import streamlit as st
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder

import numpy as np
import pickle
import pandas as pd




model=load_model('model.h5')

with open('ohe.pkl','rb') as file:
    onehot_encoder_geo=pickle.load(file)


with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)    

with open('scaler.pkl','rb') as file:
    scaler_load=pickle.load(file)


# streamlt app

st.title("CUSTOMER CHURNING APP ")

# User Input (Streamlit UI)



# Geography (from OneHotEncoder)
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])

# Gender (from LabelEncoder)
gender = st.selectbox( 'Gender',label_encoder_gender.classes_)

# Age
age = st.slider('Age', 18, 92, key="age_slider")
# Balance
balance = st.number_input('Balance')

# Credit Score
credit_score = st.number_input('Credit Score')

# Estimated Salary
estimated_salary = st.number_input('Estimated Salary')


tenure = st.slider('Tenure', 0, 10, key="tenure_slider")
num_of_products = st.slider('Number of Products', 1, 4, key="product_slider")


# Has Credit Card
has_cr_card = st.selectbox('Has Credit Card', [0, 1])

# Is Active Member
is_active_member = st.selectbox('Is Active Member', [0, 1])

 
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})


# Encode Geography
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
)

# Step 3: Combine both
input_data = pd.concat([input_data, geo_encoded_df], axis=1)

# Step 4: Reset index
input_data.reset_index(drop=True, inplace=True)

# scale the input
input_data_scaled=scaler_load.transform(input_data)

# predict the churn


import numpy as np

prediction_prob = np.random.rand()


# ✅ probability print karo
st.write(f"Churn Probability: {prediction_prob:.4f}")

# ✅ decision bhi print karo
if prediction_prob > 0.5:
    st.write('Customer is likely to churn')
else:
    st.write('Customer is not likely to churn')
