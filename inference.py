import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GradingModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def predict(model, encoder, question, ref, student, total_marks):
    text1 = f"Question: {question} Reference: {ref}"
    text2 = student

    emb1 = encoder.encode(text1, convert_to_tensor=True)
    emb2 = encoder.encode(text2, convert_to_tensor=True)

    features = torch.cat([emb1, emb2, torch.abs(emb1 - emb2)]).to(device)

    with torch.no_grad():
        score = model(features).item()

    predicted_marks = score * total_marks

    return score, predicted_marks