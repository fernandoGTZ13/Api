# models/modelo.py
from sklearn.metrics import confusion_matrix, roc_curve, auc,classification_report
from sklearn.preprocessing import label_binarize
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# Directorio donde están las imágenes
train_dir = "./PLAGAS"

# Crear un generador de datos con aumento de datos
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Cargar el modelo base preentrenado
base_model = tf.keras.applications.MobileNetV2(input_shape=(160, 160, 3),
                                               include_top=False,
                                               weights='imagenet')

# Congelar las primeras capas del modelo base
base_model.trainable = True
fine_tune_at = 150  # Ajuste para descongelar menos capas
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Crear el modelo completo
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(512, activation='relu'),  # Reducción de neuronas
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(len(train_generator.class_indices), activation='softmax')
])

# Compilar el modelo con una tasa de aprendizaje baja
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Preparar generadores con tamaño de lote reducido
batch_size = 16  # Tamaño de lote reducido
train_generator = tf.keras.preprocessing.image.ImageDataGenerator().flow_from_directory(
    train_dir,
    target_size=(160, 160),
    batch_size=batch_size,
    class_mode='categorical'
)
validation_generator = tf.keras.preprocessing.image.ImageDataGenerator().flow_from_directory(
    validation_dir, 
    target_size=(160, 160),
    batch_size=batch_size,
    class_mode='categorical'
)

# Ajustar el entrenamiento
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10**(epoch / 20))

# Entrenar el modelo
model.fit(train_generator,
          validation_data=validation_generator,
          epochs=50,
          callbacks=[early_stopping, lr_schedule])

# Después de entrenar el modelo procedemos a generar la matriz de confusión
# Generar predicciones
y_pred = model.predict(validation_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = validation_generator.classes

# Generar y mostrar la Matriz de Confusión
cm = confusion_matrix(y_true, y_pred_classes)

# Mostrar la matriz de confusión con etiquetas y una barra de color
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=validation_generator.class_indices.keys(), yticklabels=validation_generator.class_indices.keys())
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.show()

# Generar y mostrar el informe de clasificación
report = classification_report(y_true, y_pred_classes, target_names=validation_generator.class_indices.keys())
print("Classification Report:\n", report)
print('')
print('CURVA ROC')
print('')
# Calcular y mostrar la Curva ROC
n_classes = len(validation_generator.class_indices)
y_true_bin = label_binarize(y_true, classes=range(n_classes))
y_pred_bin = y_pred

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 8))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'{list(validation_generator.class_indices.keys())[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('Tasa de Falsos Positivos (False Positive Rate)')
plt.ylabel('Tasa de Verdaderos Positivos (True Positive Rate)')
plt.title('Curva ROC para Clasificación de Plagas en Cultivo de Brócoli')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# Función para predecir el tipo de plaga
def predict_image(img_path, threshold=0.7):
    img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalizar la imagen

    # Hacer la predicción
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    confidence = np.max(predictions)  # Obtener la confianza más alta

    # Mapear la clase predicha al nombre de la plaga
    class_names = list(train_generator.class_indices.keys())
    predicted_label = class_names[predicted_class[0]]

    # Comprobar si la confianza es menor que el umbral
    if confidence < threshold:
        return "Resultado no encontrado"

    # Verificar si la predicción es errónea comparando la confianza con las otras clases
    sorted_predictions = np.sort(predictions[0])[::-1]  # Ordenar predicciones en orden descendente
    second_best_confidence = sorted_predictions[1]  # Obtener la segunda mejor predicción

    # Si la diferencia entre la mejor predicción y la segunda mejor es pequeña, evitar un falso positivo
    if confidence - second_best_confidence < 0.2:
        return "Resultado no encontrado"

    return predicted_label
