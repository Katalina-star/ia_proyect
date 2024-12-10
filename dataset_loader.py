import zipfile
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def extract_and_load_data(zip_path="archive.zip", extract_path="animals", batch_size=32):
    # Descomprimir el dataset si no existe la carpeta
    if not os.path.exists(extract_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"Dataset descomprimido en: {extract_path}")
    
    # Transformaciones de las imágenes
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Redimensionar imágenes
        transforms.ToTensor(),       # Convertir a tensores
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalizar
    ])
    
    # Cargar el dataset
    dataset = datasets.ImageFolder(root=extract_path, transform=transform)
    
    # Dividir en entrenamiento y prueba
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    # Crear DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, dataset.classes