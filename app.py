import streamlit as st
from load_models import load_all_models
from inference import predict

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
student = st.text_area("Student Answer")
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