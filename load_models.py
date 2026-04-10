import os
import torch
from sentence_transformers import SentenceTransformer
from inference import GradingModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_all_models(model_dir="saved_models"):
    models = {}

    for file in os.listdir(model_dir):
        if file.endswith(".pt"):
            model_path = os.path.join(model_dir, file)

            checkpoint = torch.load(model_path, map_location=device)
            model_name = checkpoint["model_name"]

            encoder_path = os.path.join(model_dir, file.replace(".pt", "_encoder"))

            encoder = SentenceTransformer(encoder_path)

            input_size = encoder.get_embedding_dimension() * 3 # type: ignore

            model = GradingModel(input_size).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            models[model_name] = {
                "model": model,
                "encoder": encoder,
                "accuracy": checkpoint.get("accuracy", 0),
                "mse": checkpoint.get("mse", 0)
            }

    return models