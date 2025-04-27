# app.py
import streamlit as st
from PIL import Image
import io
import numpy as np
from PIL.ImageFilter import GaussianBlur


st.set_page_config(page_title="Background Remover", page_icon="🖼️", layout="centered")

st.markdown("""
    <style>
        .title {
            font-size:40px !important;
            color: #6c63ff;
            text-align: center;
            font-weight: bold;
        }
        .stButton > button {
            background-color: #6c63ff;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 16px;
        }
        .stButton > button:hover {
            background-color: #574fcf;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🧼 Remove Image Background</div>', unsafe_allow_html=True)
st.write("Upload an image and remove its background with one click.")

uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    input_image = Image.open(uploaded_file)
    st.image(input_image, caption="Original Image", use_container_width=True)

    if st.button("Remove Background"):
        with st.spinner("Processing..."):
            # Simple background removal using image processing with NumPy for better performance
            # Convert to RGBA if not already
            if input_image.mode != 'RGBA':
                input_image = input_image.convert('RGBA')
            
            # Convert to numpy array for faster processing
            img_array = np.array(input_image)
            
            # Use a simple color-based background detection (assume background is similar to corners)
            # Sample the four corners
            corners = [img_array[0,0,:3], img_array[0,-1,:3], img_array[-1,0,:3], img_array[-1,-1,:3]]
            bg_color = np.mean(corners, axis=0)
            # Calculate distance from background color for each pixel
            color_dist = np.sqrt(np.sum((img_array[...,:3] - bg_color) ** 2, axis=-1))
            # Threshold: pixels close to bg_color are background
            mask = color_dist > 35  # Lower threshold for more accurate separation
            # Morphological operations to clean up mask
            # Create alpha channel
            alpha = np.where(mask, 255, 0).astype(np.uint8)
            result = img_array.copy()
            result[..., 3] = alpha
            output_image = Image.fromarray(result)
            
            output_bytes = io.BytesIO()
            output_image.save(output_bytes, format="PNG")
            st.success("Background removed! (Note: Using simple image processing as rembg is not compatible with Python 3.13)")

            st.image(output_image, caption="Image Without Background", use_container_width=True)
            # Upscale the image to HD (2x size) using LANCZOS filter for high quality
            upscale_factor = 2
            upscaled_image = output_image.resize((output_image.width * upscale_factor, output_image.height * upscale_factor), Image.LANCZOS)
            upscaled_bytes = io.BytesIO()
            upscaled_image.save(upscaled_bytes, format="PNG")
            st.image(upscaled_image, caption="Upscaled HD Image", use_container_width=True)
            st.download_button(
                label="📥 Download HD PNG",
                data=upscaled_bytes.getvalue(),
                file_name="no_bg_hd.png",
                mime="image/png"
            )
            st.download_button(
                label="📥 Download PNG",
                data=output_bytes.getvalue(),
                file_name="no_bg.png",
                mime="image/png"
            )
