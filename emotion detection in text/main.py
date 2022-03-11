import streamlit as st
from load_model import load_model

    
encoded_dict  = {'anger':0,'fear':1, 'joy':2, 'love':3, 'sadness':4, 'surprise':5}

st.title('Emotion Detection in Texts')
st.write("")
st.write("You can detect the emotion of a sentence here. ")
st.write("")
st.write("")
st.write("")
text = st.text_input('Enter your sentence here,', '')
if text:  
    with st.spinner('Predicting the emotion...'):
        validation = load_model(text)
        dict = {}
        for key , value in zip(encoded_dict.keys(),validation[0]):
            dict.update({key:value})
        emotion = max(dict, key=dict.get)
        st.write("The predicted emotion is: ",emotion)

    st.success('Done!')

