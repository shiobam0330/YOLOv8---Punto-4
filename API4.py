from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import shutil
from pathlib import Path
import matplotlib as plt
import uvicorn
from ultralytics import YOLO
import cv2
import json

app = FastAPI()

UPLOAD_DIR = Path("Imagenes_Detectadas")
UPLOAD_DIR.mkdir(exist_ok=True)

model = YOLO('yolov8n.pt')

@app.post("/computer-vision")

async def carga_imagen(file: UploadFile = File(...)):
    try:
        file_path = UPLOAD_DIR / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        with open("resultados.json", 'r') as file:
            resultados = json.load(file)

        image = cv2.imread(str(file_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = model(image_rgb)
        detections = results[0].boxes.data.cpu().numpy()
        class_names = model.names

        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            class_name = class_names[int(cls)]   
            resultados1 = {
                 "Clase": class_name,
                 "Confianza": round(float(conf),2), 
                 "Coordenadas_1": [round(float(x1), 4), round(float(y1), 4)], 
                 "Coordenadas_2": [round(float(x2), 4), round(float(y2), 4)] 
                 }
            resultados.append(resultados1)           
            print(f"Clase: {class_name}, Confianza: {conf}, Coordenadas: ({x1}, {y1}, {x2}, {y2})")

        with open('resultados.json', 'w') as json_file:
            json.dump(resultados, json_file)
        return f"Proceso Exitoso"

    except Exception as e:
        return {"error": f"Ocurrió un error: {str(e)}"}
    finally:
        if file_path.exists():
            file_path.unlink()

@app.post("/imagen")
async def carga_imagen(file: UploadFile = File(...)):
    try:
        file_path = UPLOAD_DIR / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        image = cv2.imread(str(file_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = model(image_rgb)

        imagen_nueva = results[0].plot()
        ruta = UPLOAD_DIR / f"DETECCION_{file.filename}"
        cv2.imwrite(str(ruta), imagen_nueva)

        return FileResponse(ruta, media_type="image/jpeg")
    except Exception as e:
        return {"error": f"Ocurrió un error: {str(e)}"}
    
    finally:
        if file_path.exists():
            file_path.unlink()
            
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)