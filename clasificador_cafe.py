from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#crear el dataset generador
data_gen = ImageDataGenerator(
    rescale = 1./255, #normaliza los valores de pixeles y se transforman entre 0 y 1
    rotation_range = 10, #rota las imagentes dentro de un rango de 10 grados
    width_shift_range = 0.15, #desplaza las imagenes horizontal y verticalmente hasta un 15% del ancho o alto
    height_shift_range = 0.15, #desplaza las imagenes horizontal y verticalmente hasta un 15% del ancho o alto
    shear_range = 5, #transformaciones de corte (shear) en las imágenes, que las distorsionan ligeramente
    zoom_range = [0.7, 1.3], #se aplica zoom aleatorio a las imagenes desde 70% hasta el 130%
    validation_split = 0.2 #reservar el 20% de los datos para validacion
)

#cargar imagenes del dataset
data_gen_entrenamiento = data_gen.flow_from_directory("data_sets",
                                                      target_size = (224, 224), #Redimensiona todas las imágenes al tamaño 224x224 píxeles
                                                      batch_size = 10, #Carga las imágenes en lotes de 10
                                                      shuffle = True, #Mezclar aleatoriamente las imágenes en cada época
                                                      subset = "training", #Usar solo el 80% de las imágenes para entrenamiento.
                                                      class_mode = "categorical" #Asigna etiquetas codificadas en one-hot
                                                      )

print(data_gen_entrenamiento.class_indices)
class_indices = data_gen_entrenamiento.class_indices
clases = {v: k for k, v in class_indices.items()}

data_gen_pruebas = data_gen.flow_from_directory("data_sets",
                                                     target_size = (224,224), # Redimensiona imágenes a 224x224
                                                     batch_size = 10, #Carga las imágenes en lotes de 10
                                                     shuffle = False,  # False para mantener ordenadas las predicciones
                                                     subset = "validation", # Usa el 20% reservado para prueba.
                                                     class_mode = "categorical" # Etiquetas en formato one-hot
                                                     ) 



# for imagenes, etiquetas in data_gen_entrenamiento:
#     for i in range(10):
#         plt.subplot(2, 5, i+1)
#         plt.imshow(imagenes[i])
#         break
# plt.show()

from tensorflow.keras import layers, models

#Construcción de la red neuronal

model = models.Sequential([
    layers.Conv2D(32,(3,3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(data_gen_entrenamiento.num_classes, activation='softmax')
])

model.compile(
    optimizer = "adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("Iniciando entrenamiento...")
training = model.fit(
    data_gen_entrenamiento, 
    validation_data=data_gen_pruebas,
    epochs=10,
)
#grafica de perdida
plt.plot(training.history['accuracy'], label='Entrenamiento')
plt.plot(training.history['val_accuracy'], label='Validación')
plt.legend()
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.title('Evolución del modelo')
plt.show()


loss, accuracy = model.evaluate(data_gen_pruebas)
print(f"Pérdida: {loss:.4f}  -  Precisión: {accuracy:.4f}")

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score

predicciones_modelo = model.predict(data_gen_pruebas)
y_pred = np.argmax(predicciones_modelo, axis=1)
y_true = data_gen_pruebas.labels

acc = accuracy_score(y_true, y_pred)
print(f"Exactitud calculada: {acc:.4f}")


#Matriz de confusión
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(class_indices.keys()))
disp.plot(cmap="Blues")
plt.title("Matriz de Confusión")
plt.show()

from PIL import Image
import cv2

class_indices = data_gen_entrenamiento.class_indices
clases = {v: k for k, v in class_indices.items()}

def categorizar(ruta):
    img = Image.open(ruta)
    img = img.convert('RGB')
    img = np.array(img).astype(float)/255

    img = cv2.resize(img, (224, 224))
    img = img.reshape(-1, 224, 224, 3)
    prediccion = model.predict(img)
    indice = np.argmax(prediccion[0], axis=-1)
    return clases[indice]

ruta = "cafe_maduro.jpeg"
prediccion = categorizar(ruta)
print(prediccion)

ruta = "cafe_verde.jpg"
prediccion = categorizar(ruta)
print(prediccion)

model.save("modelo_cafe.keras")

import json
with open("clases.json", "w") as f:
    json.dump(clases, f)
