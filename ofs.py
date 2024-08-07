import streamlit as st
import pandas as pd 
import plotly.express as px 
import seaborn as sns 
import matplotlib.pyplot as plt
import numpy as np
import pickle
import streamlit as st
import time
#scikit-learn




app_mode = st.sidebar.selectbox('Select Page',['Home','Prediction']) #two pages
if app_mode=='Home':    
    


    st.title("Welcome to the Army Officers Select Prediction Application .")
    st.image("SLarmy.jpg", width=650)
    
    #image_path = "D:\\Documents\\MachineLearning02\\App01\\img.jpeg"
    #st.image(image_path)
   
    st.write("Selecting the best candidates for army officers is crucial for maintaining a strong and effective military. This involves evaluating physical fitness, leadership qualities, psychological readiness, and academic achievements.\n The Army Officers Selection Prediction Application assists in this process by using advanced machine learning algorithms to predict candidates' suitability. It is user-friendly, reliable, and provides valuable insights to support decision-making in the selection process.")
    st.write("Below diagrams and Video & audio visualize the factots of distribution in the dataset.\n\n")
    st.video("slma.mp4")
    st.audio("amio.mp3")


    st.file_uploader('Upload a  Your photo')
    df= pd.read_csv("selects.csv")
    #st.write(df)

    # Bar Plot for Outcome Counts
    st.subheader('Distribution of Army Officers Height')
    fig, ax = plt.subplots()
    sns.countplot(x='Height', data=df, ax=ax)
    for patch in ax.patches:
        current_width = patch.get_width()
        diff = current_width - 0.3
        # we change the bar width
        patch.set_width(0.3)
        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)
    ax.set_title('Distribution of Diabetes Outcomes')
    ax.set_xlabel('Height of the persons')
    ax.set_ylabel('Number of person')
    st.pyplot(fig)
    st.write("")
    st.write("")
    st.write("")

    #Age Distribution
    st.subheader('Age Distribution')
    fig, ax = plt.subplots()
    sns.histplot(df['Age'], kde= True, bins=20, ax=ax)
    ax.set_title('Age Ditribution of Individuals')
    ax.set_xlabel('Age')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)
    st.write("")
    st.write("")
    st.write("")

    #BMI Distribution
    st.subheader('BMI Distribution')
    fig, ax = plt.subplots()
    sns.histplot(df['BMI'], kde= True, bins=20, ax=ax)
    ax.set_title('BMI Ditribution of Individuals')
    ax.set_xlabel('BMI')
    ax.set_ylabel('Frequency')
    ax.set_xticklabels(['No Diabetes', 'Diabetes'])
    st.pyplot(fig)
    st.write("")
    st.write("")
    st.write("")


    #Correlation Heatmap
    st.subheader('Correlation Heatmap')
    fig, ax = plt.subplots(figsize=(10,8))
    corr = df.corr()
    sns.heatmap(corr, annot=True,  cmap ='coolwarm', ax=ax)
    ax.set_title('Correlation Heatmap of Features')
    st.pyplot(fig)
    st.write("")
    st.write("")
    st.write("")

    #Box Plot for multiple features
    st.subheader('Box Plot of Features')
    fig, ax = plt.subplots(figsize=(12,8))
    sns.boxplot(data=df[['Age', 'Gender', 'Height', 'Weight', 'BMI']], ax=ax)
    ax.set_title('Box Plot of Selected Features')
    ax.set_xlabel('Features')
    ax.set_ylabel('Value')
    st.pyplot(fig)
    st.write("")
    st.write("")
    st.write("")

    #st.figure(figsize=(10,10))
    #plt.pie(data=df['Student Population'].head(10), data=df["Name"].head(10),autopct='%1.2f%%')
    #ax.set_title('Student Population Distribution')
    #plt.show()
  
    st.sidebar.success("Select Pages From Above Menue")

elif app_mode == 'Prediction':


    loaded_model = pickle.load(open('Oselecect_model.sav', 'rb'))

    def selects_prediction(input_data):
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        prediction = loaded_model.predict(input_data_reshaped)
        print(prediction)
        if (prediction[0] == 0):
            return 'The person is not Selected'
        else:
            return 'The person is Selected'

    def main():
        st.title('Army Officers Select Prediction Application')


        age = st.text_input('Age of the Person')
        gender = st.text_input('Your Gender Male Or Female - (if Male= 1 / if Female= 0)')
        height = st.text_input('Your Height')
        weight = st.text_input('Your Weight')
        bmi = st.text_input('BMI Value')
        olevel = st.text_input( 'O/L Pass Or Fail - (if Pass= 1 / if Fail= 0)')
        alevel = st.text_input(' A/L Pass Or Fail - (if Pass= 1 / if Fail= 0)')
        
      

        diagnosis = ''
        
        if st.button('Submit'):
            with st.spinner('Please wait...'):
                time.sleep(2)
                diagnosis = selects_prediction([age, gender, height, weight, bmi, olevel, alevel ])
            
                st.success(diagnosis)

    if __name__ == '__main__':
        main()





