import torch

def load_model(model_path, model_class):
    """Loads a trained model."""
    model = model_class()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict(model, dataloader):
    """Performs prediction using the trained model."""
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            predictions.append(model(batch))
    return predictions
