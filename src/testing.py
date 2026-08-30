import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from sklearn.metrics import accuracy_score, roc_auc_score

from mAlex import mAlex
from dataloader import Dataset

# Evaluation function definition
def evaluate(model, test_loader, device):
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for i, data in enumerate(test_loader, 1):
            inputs, labels = data
            inputs = inputs.to(device)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # "occupied" class probabilities (1)
            preds = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            # Print progress every 100 batches
            if i % 100 == 0:
                print(f"Evaluating {i}/{len(test_loader)} batches...")

    accuracy = accuracy_score(all_labels, all_preds)
    auc_roc = roc_auc_score(all_labels, all_probs)

    return accuracy, auc_roc


def main():
    # Defining device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    # Setting up paths for data and model
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_root = os.path.join(project_root, "data", "CNRPark-EXT")
    model_path = os.path.join(project_root, "model", "malexnet_final.pth")

    # Transforms for the test dataset (no randomization))
    test_transforms = v2.Compose([
        v2.Resize((256, 256)),
        v2.CenterCrop(224),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Creating the Dataset for testing
    test_dataset = Dataset(
        patches_path=os.path.join(data_root, "PATCHES"),
        labels_file=os.path.join(data_root, "LABELS", "test.txt"),
        transforms=test_transforms
    )

    # Creating the DataLoader for testing
    test_loader = DataLoader(
        test_dataset, batch_size=64, shuffle=False,
        num_workers=8, drop_last=False
    )

    # Instantiating the model and loading the trained weights
    model = mAlex().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Evaluating the model on the test dataset
    accuracy, auc_roc = evaluate(model, test_loader, device)

    # Printing the evaluation results
    print(f"\nResults on the test set:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"AUC-ROC:  {auc_roc:.4f}")


if __name__ == "__main__":
    main()