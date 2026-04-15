import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# -----------------------------
# 1. Config
# -----------------------------

DATA_DIR =  "../build/data/AIDER"
num_classes = 5                # AIDER has 8 classes
BATCH_SIZE = 32
epochs = 30
IMG_SIZE = 224
lr = 1e-3
MODEL_PATH = "resnet_aider.pth"
CLASS_NAMES = ['normal', 'fire', 'flooded_area', 'traffic_accident', 'collapsed_building']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -----------------------------
# 2. Data Augmentation & Loaders
# -----------------------------
# ============ DATA LOADING ============
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
class_names = dataset.classes
num_classes = len(class_names)

# Split dataset: 50% train, 15% val, 35% test
total_len = len(dataset)
train_len = int(0.5 * total_len)
val_len = int(0.15 * total_len)
test_len = total_len - train_len - val_len

train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# -----------------------------
# 3. Model (ResNet18 pretrained on ImageNet)
# -----------------------------
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)  # replace final layer
#model = model.to(device)
res_aider = model.to(device)

# -----------------------------
# 4. Loss & Optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# -----------------------------
# 5. Training Loop
# -----------------------------
def train_model(model, train_loader, val_loader, epochs):
    best_acc = 0.0

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # --- Validation ---
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
              f"| Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "resnet18_aider_best.pth")
            print(f"Model saved to {MODEL_PATH}")

    print("Training finished. Best Val Acc:", best_acc)


train_model(model, train_loader, val_loader, epochs)
# ============ PLOTS ============
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(train_acc_hist, label='Train Acc')
plt.plot(val_acc_hist, label='Val Acc')
plt.xlabel("Epochs"); plt.ylabel("Accuracy")
plt.legend(); plt.grid()

plt.subplot(1,2,2)
plt.plot(train_loss_hist, label='Train Loss')
plt.plot(val_loss_hist, label='Val Loss')
plt.xlabel("Epochs"); plt.ylabel("Loss")
plt.legend(); plt.grid()

plt.title("training_validation_curves_50RESNet")
plt.savefig("training_validation_curves_50RESNet.png")
plt.show()

# ============ CONFUSION MATRIX ============
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        preds = outputs.argmax(1).cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(labels.numpy())

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix_#50RESNet")
plt.savefig("confusion_matrix_50RESNet.png")
plt.show()

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))
