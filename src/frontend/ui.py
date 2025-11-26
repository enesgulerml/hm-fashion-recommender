import streamlit as st
import requests
import json
import os

# --- SETTINGS ---
# Backend API adresimiz (Localhost)
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/recommend")

# Page Configuration
st.set_page_config(
    page_title="H&M AI Fashion Stylist",
    page_icon="🛍️",
    layout="wide"
)

# --- TITLE & HEADER ---
st.title("🛍️ H&M AI Personal Stylist")
st.markdown("""
**Welcome!** I'm your AI-powered style consultant. 
Describe what you're looking for in natural language, and I'll find the perfect outfit for you.
*(Example: 'I need a red dress for a wedding' or 'Sportswear for the gym')*
""")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("How many products should be brought?", min_value=1, max_value=10, value=3)
    st.info("This system uses **Semantic Search**. It looks at the MEANING of words, not just their letters.")

# --- MAIN SEARCH PART ---
query = st.text_input("What kind of outfit are you looking for?", placeholder="Summer floral dress...")
search_btn = st.button("🔎 Find My Style")

if search_btn and query:
    with st.spinner('Artificial intelligence scans the wardrobe...'):
        try:
            # Send Request to Backend
            payload = {"text": query, "top_k": top_k}
            response = requests.post(API_URL, json=payload)

            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])

                if not results:
                    st.warning("Sorry, I couldn't find anything suitable for this.")
                else:
                    st.success(f"🎉 I found {len(results)} great pieces for you!")

                    # List Results
                    for item in results:
                        with st.container():
                            col1, col2 = st.columns([1, 4])

                            with col1:
                                st.markdown("# 👗")
                                st.metric(label="Matching Score", value=f"%{item['similarity_score'] * 100:.1f}")

                            with col2:
                                st.subheader(item['product_name'])
                                st.caption(
                                    f"Category: {item['details']['product_group_name']} | Tip: {item['details']['product_type_name']}")
                                st.write(f"**Detail:** {item['details'].get('detail_desc', 'No explanation.')}")
                                st.markdown("---")

            else:
                st.error(f"An error occurred! API Code: {response.status_code}")

        except requests.exceptions.ConnectionError:
            st.error("🚨 Backend API is unavailable! Have you started the API? (uvicorn)")