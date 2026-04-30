import os
import copy
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from PIL import Image
from model import CNN
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATASET_PATH = r"E:\Project\chest_xray"
NUM_CLIENTS  = 3
NUM_ROUNDS   = 5
LOCAL_EPOCHS = 2
BATCH_SIZE   = 32
LR           = 0.001
CLASS_NAMES  = ["NORMAL", "PNEUMONIA"]

torch.manual_seed(42)
random.seed(42)

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

eval_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

def load_data():
    train_dir = os.path.join(DATASET_PATH, "train")
    test_dir  = os.path.join(DATASET_PATH, "test")
    val_dir   = os.path.join(DATASET_PATH, "val")

    for d in [train_dir, test_dir, val_dir]:
        if not os.path.isdir(d):
            print(f"[ERROR] Folder not found: {d}")
            raise SystemExit(1)

    train = datasets.ImageFolder(train_dir, transform=transform)
    test  = datasets.ImageFolder(test_dir,  transform=eval_transform)
    val   = datasets.ImageFolder(val_dir,   transform=eval_transform)

    print(f"[Dataset] Train: {len(train)}  |  Val: {len(val)}  |  Test: {len(test)}")
    return train, val, test

def split_data(dataset):
    class_indices = {}
    for idx, (_, label) in enumerate(dataset):
        class_indices.setdefault(label, []).append(idx)

    for k in class_indices:
        random.shuffle(class_indices[k])

    skew = [
        {0: 0.60, 1: 0.40},
        {0: 0.40, 1: 0.60},
        {0: 0.50, 1: 0.50},
    ]

    min_cls = min(len(v) for v in class_indices.values())
    budget  = (min_cls * 2) // NUM_CLIENTS
    ptrs    = {k: 0 for k in class_indices}
    subsets = []

    for cid in range(NUM_CLIENTS):
        indices = []
        for label, ratio in skew[cid].items():
            n     = int(budget * ratio)
            start = ptrs[label]
            end   = min(start + n, len(class_indices[label]))
            indices.extend(class_indices[label][start:end])
            ptrs[label] = end

        random.shuffle(indices)
        subsets.append(Subset(dataset, indices))

    return subsets

def train_client(model, loader):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    correct, total, total_loss = 0, 0, 0.0

    for _ in range(LOCAL_EPOCHS):
        for x, y in loader:
            optimizer.zero_grad()
            out  = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            correct    += (out.argmax(1) == y).sum().item()
            total      += y.size(0)

    return model.state_dict(), correct / total * 100, total_loss / total

def fedavg(global_model, states, sizes):
    total       = sum(sizes)
    global_dict = global_model.state_dict()
    avg_dict    = {}

    for k in global_dict:
        if global_dict[k].dtype in (torch.int64, torch.int32, torch.bool):
            avg_dict[k] = states[0][k].clone()
        else:
            avg_dict[k] = torch.zeros_like(global_dict[k], dtype=torch.float32)
            for state, size in zip(states, sizes):
                avg_dict[k] += state[k].float() * (size / total)

    global_model.load_state_dict(avg_dict)
    return global_model

def evaluate(model, loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct, total, total_loss = 0, 0, 0.0

    with torch.no_grad():
        for x, y in loader:
            out        = model(x)
            total_loss += criterion(out, y).item() * y.size(0)
            correct    += (out.argmax(1) == y).sum().item()
            total      += y.size(0)

    return correct / total * 100, total_loss / total

def federated_learning():
    train_data, val_data, test_data = load_data()

    clients     = split_data(train_data)
    loaders     = [DataLoader(c, BATCH_SIZE, shuffle=True) for c in clients]
    val_loader  = DataLoader(val_data, 64)
    test_loader = DataLoader(test_data, 64)

    model = CNN()
    history = {"round": [], "val_acc": [], "val_loss": []}

    for rnd in range(1, NUM_ROUNDS + 1):
        states, sizes = [], []

        for loader in loaders:
            local_model = copy.deepcopy(model)
            state, _, _ = train_client(local_model, loader)
            states.append(state)
            sizes.append(len(loader.dataset))

        model = fedavg(model, states, sizes)
        acc, loss = evaluate(model, val_loader)

        history["round"].append(rnd)
        history["val_acc"].append(acc)
        history["val_loss"].append(loss)

        print(f"Round {rnd}: Val Acc={acc:.2f}%")

    evaluate(model, test_loader)
    torch.save(model.state_dict(), "model.pth")

    return model, history

def predict(model, path):
    img = Image.open(path).convert("L")
    x = eval_transform(img).unsqueeze(0)

    with torch.no_grad():
        prob = torch.softmax(model(x), dim=1)
        conf, pred = prob.max(1)

    print(CLASS_NAMES[pred.item()], conf.item()*100)

def menu(model):
    while True:
        ch = input("1 NORMAL | 2 PNEUMONIA | 3 custom | exit: ")

        if ch == "exit":
            break

        elif ch == "1":
            p = os.path.join(DATASET_PATH,"test","NORMAL",
            os.listdir(os.path.join(DATASET_PATH,"test","NORMAL"))[0])
            predict(model,p)

        elif ch == "2":
            p = os.path.join(DATASET_PATH,"test","PNEUMONIA",
            os.listdir(os.path.join(DATASET_PATH,"test","PNEUMONIA"))[0])
            predict(model,p)

        elif ch == "3":
            predict(model,input("Enter path: "))

if __name__ == "__main__":
    model, history = federated_learning()
    menu(model)