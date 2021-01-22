import streamlit as st
import tensorflow
import numpy as np
import keras
from keras.applications.xception import Xception, preprocess_input
from keras.models import Model, load_model
import numpy as np
from PIL import Image
from keras.preprocessing.sequence import pad_sequences
from pickle import load

st.title("Image Caption Generator")
up_image = st.file_uploader("upload an image")


def extract_features(image, model):
    image1 = Image.open(image)
    new_image = image1.resize((299, 299))
    new_image = np.array(new_image)
    # for images that has 4 channels, we convert them into 3 channels
    if new_image.shape[2] == 4:
        new_image = new_image[..., :3]
    new_image = np.expand_dims(new_image, axis=0)
    new_image = new_image / 127.5
    new_image = new_image - 1.0
    feature = model.predict(new_image)
    return feature


def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def generate_desc(model, tokenizers, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizers.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo, sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizers)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text


if st.button('upload'):
    max_length = 32
    tokenizers = load(open("tokenizer.p", "rb"))
    model = load_model('model_9.h5')
    xception_model = Xception(include_top=False, pooling="avg")
    photo = extract_features(up_image, xception_model)
    description = generate_desc(model, tokenizers, photo, max_length)
    #up_image = Image.open(uploaded_image)
    st.image(up_image, caption='Uploaded Image.', width=250)
    description = description[5:-3]
    st.write(description)
    #st.write(up_image)
