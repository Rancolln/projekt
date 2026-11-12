import io
import os
import torch
import torch.nn as nn
import requests
import mysql.connector
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from torchvision.models import resnet50
import torchvision.transforms as transforms
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Vehicle Recognition and LLM Analytics API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Zezwala aplikacji React na komunikacje z Dockerem
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. SLOWNIKI KLAS (BEZ POLSKICH ZNAKOW) ---
KLASY = {
    0: "bicycle", 1: "boat", 2: "bus", 3: "car", 4: "helicopter",
    5: "minibus", 6: "motorcycle", 7: "taxi", 8: "train", 9: "truck"
}

NAZWY_PL = {
    "bicycle": "Rower", 
    "boat": "Lodz", 
    "bus": "Autobus", 
    "car": "Samochod",
    "helicopter": "Helikopter", 
    "minibus": "Minibus", 
    "motorcycle": "Motocykl",
    "taxi": "Taksowka", 
    "train": "Pociag", 
    "truck": "Ciezarowka"
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 2. KONFIGURACJA SIECIOWA DOCKER ---
OLLAMA_URL = "http://ollama-container:11434/api/generate"
OLLAMA_MODEL = "llama3.1" 

DB_CONFIG = {
    "host": "mysql-container",
    "user": "root",
    "password": "twoje_super_haslo", 
    "database": "autoDetektyw"
}

# --- 3. LADOWANIE MODELU RESNET50 ---
def zaladuj_model():
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)

    pelna_sciezka = "resnet50_vehicle10.pkl"

    if not os.path.exists(pelna_sciezka):
        raise FileNotFoundError(f"Brak pliku modelu: {pelna_sciezka}")

    state = torch.load(pelna_sciezka, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

model = zaladuj_model()

# --- 4. FUNKCJE INTEGRACYJNE (PROMPT BEZ ENKODOWANIA ZNAKOW) ---
def generuj_opis_ollama(pojazd_pl, pewnosc):
    prompt = (
        f"Model wizyjny rozpoznal na zdjeciu obiekt: {pojazd_pl} z pewnoscia {pewnosc}%. "
        f"Napisz krotka, profesjonalna notatke raportowa na temat rejestracji obecnosci "
        f"tego pojazdu. Odpowiedz wylacznie w 2-3 zdaniach, w jezyku polskim."
    )
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json().get("response", "Brak tekstu.")
    except Exception as e:
        return f"Nie udalo sie polaczyc z Ollama: {str(e)}"
    return "Blad generowania raportu."

def zapisz_do_mysql(id_klasy, klasa, pojazd_pl, pewnosc, raport_text):
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        query = """
            INSERT INTO raporty_pojazdow (id_klasy, klasa_en, klasa_pl, pewnosc_modelu, raport_llm)
            VALUES (%s, %s, %s, %s, %s)
        """
        cursor.execute(query, (id_klasy, klasa, pojazd_pl, pewnosc, raport_text))
        conn.commit()
        cursor.close()
        conn.close()
        print("Pryslnie zapisano wiersz w bazie MySQL.")
    except Exception as e:
        print(f"Problem z zapisem do bazy MySQL: {str(e)}")

# --- 5. ENDPOINTY ---
@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE)}

@app.post("/rozpoznaj")
async def rozpoznaj(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/webp", "image/bmp"]:
        raise HTTPException(status_code=400, detail="Nieobslugiwany format zdjecia.")

    try:
        zawartosc = await file.read()
        obraz = Image.open(io.BytesIO(zawartosc)).convert("RGB")
        tensor = TRANSFORM(obraz).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            wynik = model(tensor)
            prawdopodobienstwa = torch.softmax(wynik, dim=1)
            pewnosc, indeks = torch.max(prawdopodobienstwa, dim=1)

        id_klasy = indeks.item()
        klasa = KLASY[id_klasy]
        pojazd_pl = NAZWY_PL[klasa]
        pewnosc_procent = round(pewnosc.item() * 100, 2)

        tekst_raportu = generuj_opis_ollama(pojazd_pl, pewnosc_procent)
        zapisz_do_mysql(id_klasy, klasa, pojazd_pl, pewnosc_procent, tekst_raportu)

        return {
            "success": True,
            "class_id": id_klasy,
            "vehicle_name": pojazd_pl,
            "confidence": pewnosc_procent,
            "ollama_report": tekst_raportu,
            "saved_to_db": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Blad systemu: {str(e)}")

from agent import create_router

app.include_router(create_router(
    model=model,
    device=DEVICE,
    transform=TRANSFORM,
    klasy=KLASY,
    nazwy_pl=NAZWY_PL,
    db_config=DB_CONFIG,
    ollama_base="http://ollama-container:11434",
    ollama_model=OLLAMA_MODEL,
))