import streamlit as st
from transformers import BlipProcessor , BlipForConditionalGeneration
import imageio.v3 as iio
#-----------------------------------------------------------------------------
st.header("ERC Image Captioning and Documentation")
st.subheader("Please wait up to three minutes while the model is loading. In the meantime, feel free to watch this video")
st.video("vid.mp4", autoplay=True)
#----------------------------------------------------------------------------    
@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_blip_model()

file = st.file_uploader("Upload image to model", ["png", "jpg", "jpeg", "bmh"])
if file is not None:
    image = iio.imread(file)
    st.image(image)
    inputs = processor(images=image, return_tensors='pt')
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    st.subheader("caption for image is : ")
    st.write(caption)

    st.subheader("Try another image and tell us your opinion please , and thanks")

