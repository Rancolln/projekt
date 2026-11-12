import io
import json
import uuid
from typing import Optional

import mysql.connector
import requests
import torch
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image

SYSTEM_PROMPT = (
    "You are a vehicle-recognition assistant. You cannot see images yourself. "
    "When the user attaches an image (you will see its image_id), call recognize_vehicle "
    "to identify it. Never guess the vehicle type without calling the tool. "
    "After recognizing, if the user wants a report or asks to save it, write a short "
    "professional report (2-3 sentences) and call save_report. "
    "Use search_reports ONLY for earlier detection reports made by the recognition model. "
    "Use search_vehicles for the vehicle database (table pojazdy: vehicles, their photos and owners). "
    "Always show every row the tool returns, never summarize or drop rows. "
    "Reply in the user's language (Polish by default)."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "recognize_vehicle",
            "description": "Classify the vehicle in an uploaded image using the ResNet50 model.",
            "parameters": {
                "type": "object",
                "properties": {"image_id": {"type": "string", "description": "ID of the uploaded image"}},
                "required": ["image_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_report",
            "description": "Save a detection report to the MySQL database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "class_id": {"type": "integer"},
                    "vehicle_class": {"type": "string", "description": "English class name, e.g. car"},
                    "confidence": {"type": "number", "description": "Confidence in percent"},
                    "report": {"type": "string", "description": "Short report text"},
                },
                "required": ["class_id", "vehicle_class", "confidence", "report"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_reports",
            "description": "Search saved detection REPORTS (results of past image recognitions). Not the vehicle/owner database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "vehicle_class": {"type": "string", "description": "Optional English class name filter"},
                    "limit": {"type": "integer", "description": "Max rows, default 5"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_vehicles",
            "description": "List vehicles from the vehicle database (table pojazdy): class, photo path and owner. "
                           "Use for questions about vehicles, owners and photos.",
            "parameters": {
                "type": "object",
                "properties": {
                    "vehicle_class": {"type": "string", "description": "Optional English class name, e.g. car, truck"},
                    "owner": {"type": "string", "description": "Optional part of the owner's first or last name"},
                    "limit": {"type": "integer", "description": "Max rows, default 30"},
                },
            },
        },
    },
]


def create_router(model, device, transform, klasy, nazwy_pl, db_config, ollama_base, ollama_model):
    router = APIRouter(prefix="/agent", tags=["agent"])
    images = {}  # image_id -> PIL.Image (in memory; fine for a demo)

    # ---------- tool implementations ----------
    def recognize_vehicle(image_id: str):
        img = images.get(image_id)
        if img is None:
            return {"error": f"Unknown image_id {image_id}"}
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1)[0]
        top = torch.topk(probs, 3)
        results = [
            {
                "class_id": int(i),
                "vehicle_class": klasy[int(i)],
                "vehicle_name_pl": nazwy_pl[klasy[int(i)]],
                "confidence": round(float(p) * 100, 2),
            }
            for p, i in zip(top.values, top.indices)
        ]
        return {"best": results[0], "top3": results}

    def save_report(class_id: int, vehicle_class: str, confidence: float, report: str):
        conn = mysql.connector.connect(**db_config)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO raporty_pojazdow (id_klasy, klasa_en, klasa_pl, pewnosc_modelu, raport_llm) "
            "VALUES (%s, %s, %s, %s, %s)",
            (class_id, vehicle_class, nazwy_pl.get(vehicle_class, vehicle_class), confidence, report),
        )
        conn.commit()
        cur.close()
        conn.close()
        return {"saved": True}

    def search_reports(vehicle_class: Optional[str] = None, limit: int = 5):
        conn = mysql.connector.connect(**db_config)
        cur = conn.cursor(dictionary=True)
        q = "SELECT id, klasa_en, klasa_pl, pewnosc_modelu, raport_llm, data_dodania FROM raporty_pojazdow"
        params = []
        if vehicle_class:
            q += " WHERE klasa_en = %s"
            params.append(vehicle_class)
        q += " ORDER BY id DESC LIMIT %s"
        params.append(max(1, min(int(limit), 20)))
        cur.execute(q, params)
        rows = cur.fetchall()
        cur.close()
        conn.close()
        for r in rows:
            r["data_dodania"] = str(r["data_dodania"])
        return {"rows": rows}

    def search_vehicles(vehicle_class: Optional[str] = None, owner: Optional[str] = None, limit: int = 30):
        conn = mysql.connector.connect(**db_config)
        cur = conn.cursor(dictionary=True)
        q = "SELECT id, klasa_en, klasa_pl, nr_zdjecia, sciezka_zdjecia, wlasciciel FROM pojazdy"
        where, params = [], []
        if vehicle_class:
            where.append("klasa_en = %s")
            params.append(vehicle_class)
        if owner:
            where.append("wlasciciel LIKE %s")
            params.append(f"%{owner}%")
        if where:
            q += " WHERE " + " AND ".join(where)
        q += " ORDER BY id LIMIT %s"
        params.append(max(1, min(int(limit), 50)))
        cur.execute(q, params)
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return {"count": len(rows), "rows": rows}

    REGISTRY = {
        "recognize_vehicle": recognize_vehicle,
        "save_report": save_report,
        "search_reports": search_reports,
        "search_vehicles": search_vehicles,
    }

    def run_tool(name, args):
        fn = REGISTRY.get(name)
        if fn is None:
            return {"error": f"Unknown tool {name}"}
        try:
            return fn(**(args or {}))
        except Exception as e:
            return {"error": str(e)}

    # ---------- endpoint ----------
    @router.post("/chat")
    async def chat(
        message: str = Form(...),
        history: Optional[str] = Form(None),  # JSON list of earlier messages
        file: Optional[UploadFile] = File(None),
    ):
        messages = json.loads(history) if history else [{"role": "system", "content": SYSTEM_PROMPT}]

        user_text = message
        if file is not None:
            if file.content_type not in ["image/jpeg", "image/png", "image/webp", "image/bmp"]:
                raise HTTPException(status_code=400, detail="Unsupported image format.")
            img = Image.open(io.BytesIO(await file.read())).convert("RGB")
            image_id = uuid.uuid4().hex[:8]
            images[image_id] = img
            user_text += f"\n\n[Attached image, image_id={image_id}]"
        messages.append({"role": "user", "content": user_text})

        tool_trace = []
        for _ in range(5):  # max 5 reasoning/tool steps
            try:
                r = requests.post(
                    f"{ollama_base}/api/chat",
                    json={"model": ollama_model, "messages": messages, "tools": TOOLS, "stream": False},
                    timeout=120,
                )
                r.raise_for_status()
            except Exception as e:
                raise HTTPException(status_code=502, detail=f"Ollama error: {e}")

            msg = r.json()["message"]
            messages.append(msg)
            calls = msg.get("tool_calls")
            if not calls:
                return {"reply": msg.get("content", ""), "tool_calls": tool_trace, "history": messages}

            for c in calls:
                name = c["function"]["name"]
                args = c["function"].get("arguments", {})
                result = run_tool(name, args)
                tool_trace.append({"tool": name, "args": args, "result": result})
                messages.append({"role": "tool", "tool_name": name, "content": json.dumps(result, ensure_ascii=False)})

        return {"reply": "Agent stopped: too many steps.", "tool_calls": tool_trace, "history": messages}

    return router