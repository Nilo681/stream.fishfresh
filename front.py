
import tensorflow.keras
import numpy as np

from streamlit.elements.image import image_to_url
from PIL import Image ,ImageOps

import streamlit as st

st.set_page_config(
    page_title="Image Classification App",
    page_icon="🐟",
    layout="wide",
    #initial_sidebar_state="expanded",
)
#st.image('top.png')
st.title("Analizador de frescor do pescado")

#st.sidebar.subheader("Input")

#model = ['/stream.fishfresh/model/keras_model.h5']

# component to upload images
img = st.file_uploader("Carregue uma imagem", type=["jpg", "png"])
if img:
    image = Image.open(img)
    #st.image(image)
    #st.text(type(image))

    ######## modelo##########

    np.set_printoptions(suppress=True)

    # Load the model
    model = tensorflow.keras.models.load_model('saved_model.h5')
    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    #image = Image.open('image')
    #image = ImageOps.open(img)


    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    st.image(image)
    #st.text(type(image))
    #turn the image into a numpy array
    image_array = np.asarray(image)
    # display the resized image
    #image.show()

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    classes = ['Dia-1','Dia-2','Dia-3','Dia-4','Dia-6','Dia-7','Dia-8','Dia-9']# run the inference
    prediction = model.predict(data)


    maior = np.argmax(prediction)

    predicted_class = classes[maior]
    st.text(predicted_class)
    print(predicted_class)
    #return predicted_class
