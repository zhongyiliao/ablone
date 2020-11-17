import streamlit as st
import pandas as pd
import pickle

st.write("""
### Hello!
""")

st.sidebar.header('User Input')
st.sidebar.subheader('Please enter your data:')



def get_input():
    #widgets
    v_Sex = st.sidebar.radio('Sex', ['Male','Female','Infant'])
    v_Length = st.sidebar.slider('Length', 0.075, 0.745, 0.506)
    v_Height = st.sidebar.slider('Height', 0.055, 0.60, 0.4)
    v_Diameter = st.sidebar.slider('Diameter', 0.01, 0.24, 0.13)
    v_Whole_weight = st.sidebar.slider('Whole Weight', 0.002, 2.55, 0.78)
    v_Shucked_weight = st.sidebar.slider('Shucked Weight', 0.001, 1.07, 0.3)
    v_Viscera_weight = st.sidebar.slider('Viscera Weight', 0.0005, 0.54, 0.17)
    v_Shell_weight = st.sidebar.slider('Shell Weight', 0.001, 1.005, 0.24)
    

    if v_Sex == 'Male': v_Sex = 'M'
    elif v_Sex == 'Female': v_Sex = 'F'
    else: v_Sex = 'I'

    #dictionary
    data = {'Sex': v_Sex,
            'Length': v_Length,
            'Height': v_Height,
            'Diameter': v_Diameter,
            'Whole_weight': v_Whole_weight,
            'Shucked_weight': v_Shucked_weight,
            'Viscera_weight': v_Viscera_weight,
            'Shell_weight': v_Shell_weight}

    #create data frame
    data_df = pd.DataFrame(data, index=[0])
    return data_df

df = get_input()
st.write(df)

data_sample = pd.read_csv('abalone_sample_data.csv')
df = pd.concat([df, data_sample],axis=0)

cat_data = pd.get_dummies(df[['Sex']])

#Combine all transformed features together
X_new = pd.concat([cat_data, df], axis=1)
X_new = X_new[:1] # Select only the first row (the user input data)
#Drop un-used feature
X_new = X_new.drop(columns=['Sex'])


# -- Reads the saved normalization model
load_nor = pickle.load(open('normalization.pkl', 'rb'))
#Apply the normalization model to new data
X_new = load_nor.transform(X_new)
st.write(X_new)

# -- Reads the saved classification model
load_knn = pickle.load(open('best_knn.pkl', 'rb'))
# Apply model for prediction
prediction = load_knn.predict(X_new)
st.write(prediction)