import streamlit as st
import requests
from PIL import Image

def main():
    st.title("Triton Instance Segmentation")

    uploaded_image = st.file_uploader("Upload Image", type=['jpg', 'png'])

    if uploaded_image is not None:
        st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

        if st.button('Process'):
            response = requests.post('http://localhost:5000/infer', files={'image': uploaded_image})
            if response.status_code == 200:
                output_data = response.json()['output']
                # Process output_data as needed
                st.write("Inference Output:")
                st.json(output_data)
            else:
                st.error("Error processing image")

if __name__ == '__main__':
    main()
import streamlit as st
import requests
from PIL import Image

def main():
    st.title("Triton Instance Segmentation")

    uploaded_image = st.file_uploader("Upload Image", type=['jpg', 'png'])

    if uploaded_image is not None:
        st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

        if st.button('Process'):
            response = requests.post('http://localhost:5000/infer', files={'image': uploaded_image})
            if response.status_code == 200:
                output_data = response.json()['output']
                # Process output_data as needed
                st.write("Inference Output:")
                st.json(output_data)
            else:
                st.error("Error processing image")

if __name__ == '__main__':
    main()
import streamlit as st
import requests
from PIL import Image

def main():
    st.title("Triton Instance Segmentation")

    uploaded_image = st.file_uploader("Upload Image", type=['jpg', 'png'])

    if uploaded_image is not None:
        st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

        if st.button('Process'):
            response = requests.post('http://localhost:5000/infer', files={'image': uploaded_image})
            if response.status_code == 200:
                output_data = response.json()['output']
                # Process output_data as needed
                st.write("Inference Output:")
                st.json(output_data)
            else:
                st.error("Error processing image")

if __name__ == '__main__':
    main()
