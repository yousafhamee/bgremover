# app.py
import streamlit as st
from PIL import Image
import io
import numpy as np
from PIL.ImageFilter import GaussianBlur
from scipy import ndimage
from sklearn.cluster import KMeans

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

def get_kmeans_mask(img_array, n_clusters=3, bg_sample_coords=None):
    h, w, c = img_array.shape
    flat_img = img_array[...,:3].reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_clusters, n_init=5, random_state=42)
    labels = kmeans.fit_predict(flat_img)
    label_img = labels.reshape(h, w)
    # If not provided, sample corners for background
    if bg_sample_coords is None:
        bg_sample_coords = [(0,0), (0,w-1), (h-1,0), (h-1,w-1)]
    bg_labels = [label_img[y, x] for y, x in bg_sample_coords]
    # Most common label among corners is background
    from collections import Counter
    bg_label = Counter(bg_labels).most_common(1)[0][0]
    mask = label_img != bg_label
    return mask

if uploaded_file:
    input_image = Image.open(uploaded_file)
    st.image(input_image, caption="Original Image", use_container_width=True)

    if st.button("Remove Background"):
        with st.spinner("Processing..."):
            if input_image.mode != 'RGBA':
                input_image = input_image.convert('RGBA')
            img_array = np.array(input_image)
            # Use k-means clustering for better background/foreground separation
            mask = get_kmeans_mask(img_array, n_clusters=3)
            # Refine mask with morphological operations
            mask = ndimage.binary_dilation(mask, iterations=2)
            mask = ndimage.binary_erosion(mask, iterations=2)
            # Optionally, edge detection to further refine
            edges = ndimage.sobel(mask.astype(float))
            mask = np.logical_or(mask, edges > 0.1)
            alpha = np.where(mask, 255, 0).astype(np.uint8)
            result = img_array.copy()
            result[..., 3] = alpha
            output_image = Image.fromarray(result)
            output_bytes = io.BytesIO()
            output_image.save(output_bytes, format="PNG")
            st.success("Background removed! (Improved for complex images using k-means clustering)")
            st.image(output_image, caption="Image Without Background", use_container_width=True)
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
