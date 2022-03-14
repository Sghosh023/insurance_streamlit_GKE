#from pycaret.regression import load_model, predict_model
from catboost import CatBoostRegressor
import streamlit as st
import pandas as pd
import numpy as np

#print(pd.__version__)

#model = load_model('deployment_28042020')
cb_model = CatBoostRegressor()
cb_model.load_model("model.json","json")

def predict(input_df):
    #predictions_df = predict_model(estimator=model, data=input_df)
    #predictions = predictions_df['Label'][0]
    predictions = cb_model.predict(input_df)
    return np.round(predictions,2)

def run():

    from PIL import Image
    #image = Image.open('logo.png')
    image_hospital = Image.open('hospital.jpg')

    #st.image(image,use_column_width=False)

    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))

    st.sidebar.info('This app is created to predict Insurance Premium amount')
    #st.sidebar.success('https://www.pycaret.org')
    
    st.sidebar.image(image_hospital)

    st.title("Insurance Premium Prediction App")

    if add_selectbox == 'Online':

        age = st.number_input('Age', min_value=10, max_value=100, value=25)
        sex = st.selectbox('Sex', ['male', 'female'])
        bmi = st.number_input('BMI', min_value=10, max_value=50, value=10)
        children = st.selectbox('Children', [0,1,2,3,4,5,6,7])
        if st.checkbox('Smoker'):
            smoker = 'yes'
        else:
            smoker = 'no'
        region = st.selectbox('Region', ['southwest', 'northwest', 'northeast', 'southeast'])

        output=""

        input_dict = {'age' : age, 'sex' : sex, 'bmi' : bmi, 'children' : children, 'smoker' : smoker, 'region' : region}
        input_df = pd.DataFrame([input_dict])

        if st.button("Predict"):
            output = predict(input_df=input_df)
            output = '$' + str(output)

        st.success('The output is {}'.format(output))

    if add_selectbox == 'Batch':

        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            list_of_columns = ['age', 'sex', 'bmi', 'children', 'smoker', 'region' ]
            if list(data.columns) == list_of_columns:
                result = predict(input_df=data)
                st.write(result)
            else:
                st.write("Please upload a valid csv file containing only six columns('age', 'sex', 'bmi', 'children', 'smoker', 'region')")

if __name__ == '__main__':
    run()