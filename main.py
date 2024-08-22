# main.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import threading
import uvicorn

from models.modelo import predict_image

app = FastAPI()

# URL para saber si la API está activa
@app.get("/")
def read_root():
    return {"message": "¡Bienvenido a la API REST con FastAPI 31!"}

# Método para recibir la imagen
@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Leer el archivo subido
        contents = await file.read()

        # Convertir los bytes en una imagen utilizando PIL
        img = Image.open(BytesIO(contents))

        # Guardar la imagen en el sistema de archivos
        img_path = "static/decoded_image.png"
        img.save(img_path)

        # Predecir la clase de la imagen
        prediccion = predict_image(img_path)

        # Mostrar la imagen de forma no bloqueante
        # plt.imshow(img)
        # plt.axis('off')  # Opcional: para ocultar los ejes
        # plt.show(block=False)

        return {"message": f"{prediccion}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Ejecutar Uvicorn en segundo plano en el puerto 8000
def run():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    threading.Thread(target=run).start()
