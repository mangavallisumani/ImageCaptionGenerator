import numpy as np
from PIL import Image
import streamlit as st
from keras.applications.vgg16 import VGG16
from keras.models import Model, load_model
from pickle import load
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input

st.title("Image Caption Generator")
up_image = st.file_uploader("upload an image")


def extract_features(filename):
    model = VGG16()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    image1 = Image.open(filename)
    image = image1.resize((224, 224))
    image = img_to_array(image)
    image = image.reshape((1, 224, 224, 3))  # batch size height width channels
    image = preprocess_input(image)
    feature = model.predict(image)
    return feature


def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo, sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text


if st.button('upload'):
    max_length = 34
    tokenizers = load(open("tokenizer.p", "rb"))
    model = load_model('modelnew_9.h5')
    photo = extract_features(up_image)
    description = generate_desc(model, tokenizers, photo, max_length)
    st.image(up_image, caption='Uploaded Image.', width=250)
    description = description[5:-3]
    st.write(description)

