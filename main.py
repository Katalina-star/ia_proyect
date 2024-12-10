import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch
from dataset_loader import extract_and_load_data
from model_definition import CNN  # Usamos la clase que implementa ResNet18
from train import train_model
from evaluate import evaluate_model

# Función para mostrar imágenes
def imshow(img):
    img = img / 2 + 0.5  # Desnormalizar
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Definir variables
zip_path = 'archive.zip'      # Archivo zip con las imágenes
extract_path = 'animals'      # Carpeta donde están las imágenes descomprimidas
batch_size = 32               # Tamaño de los lotes

# Cargar el conjunto de datos
train_loader, test_loader, classes = extract_and_load_data(zip_path, extract_path, batch_size)

# Verificar si los datos se cargaron correctamente
if train_loader is None or len(classes) == 0:
    print("Error al cargar los datos o las clases")
else:
    # Obtener un lote de entrenamiento para visualización
    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    # Mostrar algunas imágenes de ejemplo
    imshow(torchvision.utils.make_grid(images))
    print('Clases en las imágenes mostradas:', ' '.join(f'{classes[labels[j]]}' for j in range(batch_size)))

    # Definir el modelo con el número de clases
    num_classes = len(classes)
    model = CNN(num_classes)  # Usar la clase que implementa ResNet18

    # Entrenar el modelo
    print("\nIniciando entrenamiento...\n")
    train_model(model, train_loader, epochs=50, lr=0.0001, device="cpu")

    # Evaluar el modelo en el conjunto de prueba
    print("\nEvaluando el modelo...\n")
    evaluate_model(model, test_loader, device="cpu")

    # Guardar el modelo entrenado
    torch.save(model.state_dict(), "animal_classifier.pth")
    print("\nModelo guardado como 'animal_classifier.pth'.")
