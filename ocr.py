import ollama

def extract_text_from_image(image_path):
    try:
        response = ollama.chat(
            model="glm-ocr:latest",
            messages=[
                {
                    "role": "user",
                    "content": "Extract all text from this image. Don't add extra words.",
                    "images": [image_path]
                }
            ]
        )

        return response["message"]["content"]

    except Exception as e:
        return f"OCR Error: {str(e)}"