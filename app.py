import streamlit as st
from load_models import load_all_models
from inference import predict
import tempfile
from ocr import extract_text_from_image

st.set_page_config(page_title="Model Evaluator", layout="wide")

st.title("🧠 Answer Grading Model Comparator")

# Load models once
@st.cache_resource
def load_models_cached():
    return load_all_models()

models = load_models_cached()

# -------------------------
# Input Section
# -------------------------
st.header("✏️ Input")

question = st.text_area("Question")
reference = st.text_area("Reference Answer")
st.subheader("📄 Student Answer")

upload_option = st.radio(
    "Choose input method:",
    ["Type Answer", "Upload Image (OCR)"]
)

student = ""

if upload_option == "Type Answer":
    student = st.text_area("Student Answer")

else:
    uploaded_file = st.file_uploader("Upload Answer Image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Save temp image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        if st.button("🧠 Extract Text (OCR)"):
            with st.spinner("Running OCR..."):
                extracted_text = extract_text_from_image(temp_path)

            st.success("OCR Complete ✅")

            student = extracted_text

            st.text_area("Extracted Answer", value=student, height=200)
total_marks = st.number_input("Total Marks", min_value=1.0, value=10.0)

run = st.button("🚀 Evaluate")

# -------------------------
# Output Section
# -------------------------
if run:
    if not question or not reference or not student:
        st.warning("Please fill all fields")
    else:
        st.header("📊 Results")

        results = []

        for name, data in models.items():
            score, marks = predict(
                data["model"],
                data["encoder"],
                question,
                reference,
                student,
                total_marks
            )

            results.append({
                "Model": name,
                "Score (0-1)": round(score, 3),
                "Predicted Marks": round(marks, 2),
                "Train Accuracy": round(data["accuracy"], 3),
                "Train MSE": round(data["mse"], 2)
            })

        # Sort by predicted marks
        results = sorted(results, key=lambda x: x["Predicted Marks"], reverse=True)

        st.dataframe(results)

        # Highlight best
        best = results[0]
        st.success(f"🏆 Best Model: {best['Model']} → {best['Predicted Marks']} marks")