import os
import torch
import torch.nn as nn
from mAlex import mAlex
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from dataloader import Dataset

# Training function definition
def train(epoch, train_loader, model, loss_fn, device):
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
    print_every = max(1, len(train_loader) // 20)

    # Loop over epochs
    for ep in range(epoch):
        if ep >= 12:
            learning_rate = 0.0025
        elif ep >= 6:
            learning_rate = 0.005
        else:
            learning_rate = 0.01

        optimizer.param_groups[0]['lr'] = learning_rate
        running_loss = 0.0

        print("Epoch {}".format(ep+1))

        # Loop over batches
        for i, data in enumerate(train_loader, 1):
            # Mooving input and labels to the device
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Print loss every few batches (73 batches for 64 batch size)
            if i % print_every == 0:
                print('Epoch {}.\tBatch {}/{}.\tLoss = {:.3f}.'.format(
                    ep + 1, i, len(train_loader), running_loss / print_every))
                running_loss = 0.0
        print('Finished Epoch {}.'.format(ep + 1))
    print('Finished Training.')


def main():
    # Defining device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device} device")

    # Defining transforms for the dataset
    transforms = v2.Compose([
        v2.Resize((256, 256)),
        v2.RandomCrop(224),
        v2.RandomHorizontalFlip(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Setting up paths for data and model
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_root = os.path.join(project_root, "data", "CNRPark-EXT")
    model_dir = os.path.join(project_root, "model")

    # Creating the Dataset for training
    dataset = Dataset(
        patches_path=os.path.join(data_root, "PATCHES"),
        labels_file=os.path.join(data_root, "LABELS", "train.txt"),
        transforms=transforms
    )

    # Instantiating the model and moving it to the device
    model = mAlex().to(device)

    # Defining the loss function
    loss_fn = nn.CrossEntropyLoss()

    # Creating the DataLoader for training
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8, drop_last=False)

    # Actual training process (18 epochs)
    train(18, train_loader, model, loss_fn, device)

    # Saving the trained model
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, "malexnet_final.pth"))

if __name__ == "__main__":
    main()