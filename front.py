
import tensorflow.keras
import numpy as np

from streamlit.elements.image import image_to_url
from PIL import Image ,ImageOps

import streamlit as st

from pathlib import Path
import base64

st.set_page_config(
    page_title="Image Classification App",
    page_icon="🐟",
    layout="wide",
    #initial_sidebar_state="expanded",
)

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

#st.image('top.png')
st.title("Analizador de frescor do peixe")
st.subheader("O objetivo desta aplicação é avaliar a quanto tempo ele foi capturado e está no gelo")
st.subheader('1-clique em browser file e escolha a foto ou tire uma foto.')
st.subheader('2-aguarde a mensagem de sucesso')
st.subheader('3-veja a quantos dias o seu peixo foi capturado.')

#st.sidebar.subheader('EXEMPLOOOOOOOOOOO') Exemplo de side bar, basta descomentar

#model = ['/stream.fishfresh/model/keras_model.h5']

# component to upload images
#img = st.file_uploader("Carregue uma imagem", type=["jpg", "png"])
img = st.file_uploader(
    "Carregue uma imagem ou tire uma foto", type=["jpg", "jpeg", "png"])
if img:
    image = Image.open(img)
    #im = Image.open("Ba_b_do8mag_c6_big.png")
    image = image.convert('RGB')
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

    #classes = ['1','2','3','4','6','7','8','9']# run the inference
    classes  = ['Este peixe foi capturado à 1 Dia.',
                'Este peixe foi capturado à 2 Dias.',
                'Este peixe foi capturado à 3 Dias.',
                'Este peixe foi capturado à 4 Dias.',
                'Este peixe foi capturado à 6 Dias.',
                'Este peixe foi capturado à 7 Dias.',
                'Este peixe foi capturado à 8 Dias.',
                'Este peixe foi capturado à 9 Dias.']
    prediction = model.predict(data)


    maior = np.argmax(prediction)
    predicted_class = classes[maior]
    st.success(predicted_class)
    print(predicted_class)
    #return predicted_class
st.markdown('')
st.markdown('')
st.markdown('')
st.markdown('')
st.markdown('''[<img src='data:image/png;base64,{}' class='img-fluid' width=32 height=32>](https://www.instagram.com/lab.tecaap/) <small>Powered by llaboratory TECAAP.</small>'''.format(img_to_bytes("instagram.png")), unsafe_allow_html=True)    
st.markdown('''TECAAP 2021''')   

