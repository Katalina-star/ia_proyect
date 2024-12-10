import torch
import torch.optim as optim
from torch import nn

def train_model(model, train_loader, epochs=10, lr=0.0001, device="cpu"):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Reiniciar gradientes
            optimizer.zero_grad()
            
            # Forward + backward + optimización
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        print(f"Época {epoch+1}/{epochs}, Pérdida: {running_loss/len(train_loader):.4f}")
