#Benjamin Dong 
#Apex Club Hackathon Project 
#Skin Analysis

#Imports 
from matplotlib import transforms
import torch 
import torch.nn as nn 
import torch.optim as optim
import os
import matplotlib.pyplot as plt 

#From Imports
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models 
from datasets import load_dataset 
from sklearn.metrics import classification_report 
from PIL import Image 
from tqdm import tqdm 
#from utils import transform_example, SimpleCNN, id2label 

#Configuration of NN
MODEL_DIR = "saved_model"
BATCH_SIZE = 32
EPOCHS = 2
LR = 1e-4
IMG_SIZE = 128
NUM_CLASSES = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Using device: {device}")

#Loading Datasets 
#HuggingFace's Dataset
print("Loading dataset...")
# train_dataset = load_dataset("tbuckley/synthetic-derm-1M", split="train[:80%]")
# val_dataset = load_dataset("tbuckley/synthetic-derm-1M", split="train[80%:90%]")
# test_dataset = load_dataset("tbuckley/synthetic-derm-1M", split="train[90%:]")
# Load random 5,000 images Problem: It downloads ful 1M first and then splits them 
# train_dataset = load_dataset("tbuckley/synthetic-derm-1M", split="train").shuffle(seed=42).select(range(500))
# val_dataset = load_dataset("tbuckley/synthetic-derm-1M", split="train").shuffle(seed=42).select(range(500, 750))
# test_dataset = load_dataset("tbuckley/synthetic-derm-1M", split="train").shuffle(seed=42).select(range(750, 1000))

# dataset_sample = load_dataset("tbuckley/synthetic-derm-1M", split="train[:1000]").shuffle(seed=42)

# train_dataset = dataset_sample.select(range(0, 800))
# val_dataset = dataset_sample.select(range(800, 900))
# test_dataset = dataset_sample.select(range(900, 1000))


dataset = load_dataset("tbuckley/synthetic-derm-1M", streaming=True)
dataset_sample = dataset.take(1000)  # Takes first 1000 without full download

# Convert to regular dataset for easier use
dataset_sample = dataset_sample.shuffle(seed=42, buffer_size=1000)
train_dataset = list(dataset_sample.take(800))
val_dataset = list(dataset_sample.skip(800).take(100)) 
test_dataset = list(dataset_sample.skip(900).take(100))

#Number oF unique Classes Dynamically 
labels_list = train_dataset.features["label"].names
NUM_CLASSES = len(labels_list)
print(f"Found {NUM_CLASSES} classes.")

#Preprocessing of Image
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
]) 


def transform_example(example) :
    example["pixel_value"] = transform(example["pixel_value"])
    return example

train_dataset = train_dataset.map(transform_example)
val_dataset = val_dataset.map(transform_example)
test_dataset = test_dataset.map(transform_example)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


#Model
print("Building model...")
model = models.resnet18(pretrained=True)
# Modify the final fully connected layer
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(device)

#Training Setup 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

#Setup for Training Model on Google Colab 
print("Starting training...")


for epoch in range(EPOCHS):
    model.train()
    total_loss, correct, total = 0, 0,0 

#Training Loop
    for batch in train_loader:
        inputs = batch["pixel_value"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct/total
    avg_loss = total_loss/len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_loss:.4f} - Train Acc: {train_acc:.4f}")

    #Data Validation
    model.eval()
    all_preds, all_labels = [], []

    # Validation Loop
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["pixel_value"].to(device)
            labels = batch["label"]

            outputs = model(inputs)
            preds = outputs.argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    # val_acc = 100 * (np.array(all_preds) == np.array(all_labels)).mean()
    print("Validation Report:")
    print(classification_report(all_labels, all_preds, target_names=labels_list))
    #print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}, Val Accuracy: {val_acc:.2f}%")

#Save
os.makedirs(MODEL_DIR, exist_ok=True)
torch.save(model.state_dict(), f"{MODEL_DIR}/skin_classifier.pth")
print(f"Training Complete! Model saved to {MODEL_DIR}/skin_classifier.pth")

#Sample Predictions
print("Generating sample predictions...")
model.eval()
images, true_labels, pred_labels = [], [], []

with torch.no_grad():
    for batch in test_loader:
        inputs = batch["pixel_value"].to(device)
        labels = batch["label"]

        outputs = model(inputs)
        preds = outputs.argmax(dim=1).cpu()

        for i in range(min(4, len(inputs))):
            images.append(inputs[i].cpu().numpy())
            true_labels.append(labels[i].item())
            pred_labels.append(preds[i].item())
        break

fig, axes = plt.subplots(2, 2, figsize=(16, 8))
for i in range(len(images)):
    ax = axes[i//2, i%2]
    ax.imshow(images[i].permute(1, 2, 0))
    ax.set_title(f"True: {true_labels[i]}, Pred: {labels_list[pred_labels[i]]}")
    ax.axis("off")
plt.tight_layout()
plt.savefig(f"{MODEL_DIR}/sample_predictions.png")
print(f"Predictions sample saved to {MODEL_DIR}/sample_predictions.png")
