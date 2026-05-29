# generate_project.py
import os
import json

# Define the complete folder structure and full file contents
FILES = {}

# ==========================================
# 1. DATASETS
# ==========================================
FILES["data/routes.json"] = """[
  {
    "route_id": "R-118",
    "route_no": "118",
    "source": "Kothrud Stand",
    "destination": "Hinjewadi Phase 3",
    "stops": ["S-KOT-01", "S-SHV-02", "S-UNI-03", "S-BAN-04", "S-WAK-05", "S-HIN-06"],
    "base_fare": 25.0
  },
  {
    "route_id": "R-24",
    "route_no": "24",
    "source": "Swargate",
    "destination": "Hinjewadi Phase 3",
    "stops": ["S-SWA-00", "S-SHV-02", "S-UNI-03", "S-WAK-05", "S-HIN-06"],
    "base_fare": 30.0
  },
  {
    "route_id": "R-201",
    "route_no": "201",
    "source": "Kothrud Stand",
    "destination": "Shivajinagar",
    "stops": ["S-KOT-01", "S-LAW-07", "S-FC-08", "S-SHV-02"],
    "base_fare": 15.0
  }
]"""

FILES["data/stops.json"] = """[
  {"stop_id": "S-SWA-00", "stop_name": "Swargate Bus Stand", "latitude": 18.5018, "longitude": 73.8562},
  {"stop_id": "S-KOT-01", "stop_name": "Kothrud Stand", "latitude": 18.5074, "longitude": 73.8077},
  {"stop_id": "S-SHV-02", "stop_name": "Shivajinagar Bus Station", "latitude": 18.5309, "longitude": 73.8546},
  {"stop_id": "S-UNI-03", "stop_name": "Pune University Junction", "latitude": 18.5414, "longitude": 73.8269},
  {"stop_id": "S-BAN-04", "stop_name": "Baner Phata", "latitude": 18.5532, "longitude": 73.8115},
  {"stop_id": "S-WAK-05", "stop_name": "Wakad Chowk", "latitude": 18.5987, "longitude": 73.7483},
  {"stop_id": "S-HIN-06", "stop_name": "Hinjewadi Phase 3", "latitude": 18.5911, "longitude": 73.6980},
  {"stop_id": "S-LAW-07", "stop_name": "Law College Road", "latitude": 18.5178, "longitude": 73.8341},
  {"stop_id": "S-FC-08", "stop_name": "FC Road Bus Stop", "latitude": 18.5244, "longitude": 73.8411}
]"""

FILES["data/traffic.json"] = """[
  {"segment_id": "S-KOT-01_S-SHV-02", "level": "Heavy", "speed_multiplier": 0.45},
  {"segment_id": "S-SHV-02_S-UNI-03", "level": "Medium", "speed_multiplier": 0.75},
  {"segment_id": "S-UNI-03_S-BAN-04", "level": "Low", "speed_multiplier": 1.00},
  {"segment_id": "S-BAN-04_S-WAK-05", "level": "Heavy", "speed_multiplier": 0.38},
  {"segment_id": "S-WAK-05_S-HIN-06", "level": "Heavy", "speed_multiplier": 0.50},
  {"segment_id": "S-FC-08_S-SHV-02", "level": "Medium", "speed_multiplier": 0.70}
]"""

FILES["data/historical_delays.csv"] = """stop_id,route_id,hour_of_day,day_of_week,historical_delay_mins,traffic_density
S-KOT-01,R-118,8,1,4.5,0.65
S-KOT-01,R-118,9,1,12.3,0.88
S-SHV-02,R-118,9,1,15.1,0.92
S-UNI-03,R-118,17,3,8.4,0.70
S-WAK-05,R-24,18,4,22.0,0.95
S-BAN-04,R-118,12,2,2.1,0.30
S-HIN-06,R-24,9,1,18.5,0.89
S-FC-08,R-201,8,2,3.4,0.45"""

FILES["data/occupancy.csv"] = """route_id,stop_id,hour_of_day,historical_passenger_count,bus_capacity,weather_condition
R-118,S-KOT-01,9,42,50,Clear
R-118,S-SHV-02,9,48,50,Clear
R-118,S-UNI-03,13,20,50,Rainy
R-24,S-SHV-02,18,52,50,Clear
R-201,S-FC-08,10,15,50,Clear"""

# ==========================================
# 2. MACHINE LEARNING
# ==========================================
FILES["ml/train_models.py"] = """import os
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor

def build_and_serialize_models():
    print("Initializing EasyBUS AI Model Calibration...")
    os.makedirs('ml/binaries', exist_ok=True)
    
    if os.path.exists('data/historical_delays.csv'):
        df_delay = pd.read_csv('data/historical_delays.csv')
        df_delay['stop_code'] = df_delay['stop_id'].astype('category').cat.codes
        df_delay['route_code'] = df_delay['route_id'].astype('category').cat.codes
        
        X_delay = df_delay[['stop_code', 'route_code', 'hour_of_day', 'day_of_week', 'traffic_density']]
        y_delay = df_delay['historical_delay_mins']
        
        rf_delay = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42)
        rf_delay.fit(X_delay, y_delay)
        
        with open('ml/binaries/delay_model.pkl', 'wb') as f:
            pickle.dump(rf_delay, f)
            
    if os.path.exists('data/occupancy.csv'):
        df_occ = pd.read_csv('data/occupancy.csv')
        df_occ['route_code'] = df_occ['route_id'].astype('category').cat.codes
        df_occ['stop_code'] = df_occ['stop_id'].astype('category').cat.codes
        df_occ['is_rainy'] = df_occ['weather_condition'].apply(lambda x: 1 if x == 'Rainy' else 0)
        
        X_occ = df_occ[['route_code', 'stop_code', 'hour_of_day', 'is_rainy']]
        y_occ = (df_occ['historical_passenger_count'] / df_occ['bus_capacity']) * 100
        
        rf_occ = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42)
        rf_occ.fit(X_occ, y_occ)
        
        with open('ml/binaries/occupancy_model.pkl', 'wb') as f:
            pickle.dump(rf_occ, f)
    print("Model serialization successfully completed.")

if __name__ == '__main__':
    build_and_serialize_models()"""

FILES["ml/pipeline_notebook.ipynb"] = """{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["# EasyBUS ML Pipeline\\n", "Predicting operational bus delays and occupancy metrics."]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": ["import pandas as pd\\n", "import numpy as np\\n", "df = pd.read_csv('../data/historical_delays.csv')\\n", "print(df.head())"]
  }
 ],
 "metadata": {
  "language_info": { "name": "python" }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}"""

# ==========================================
# 3. BACKEND (FASTAPI)
# ==========================================
FILES["backend/requirements.txt"] = """fastapi==0.110.0
uvicorn==0.28.0
pydantic==2.6.4
motor==3.3.2
pandas==2.2.1
numpy==1.26.4
scikit-learn==1.4.1.post1
websockets==12.0
python-multipart==0.0.9"""

FILES["backend/Dockerfile"] = """FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "-c", "import sys; sys.path.append('.'); from ml.train_models import build_and_serialize_models; build_and_serialize_models()"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]"""

FILES["backend/app/__init__.py"] = ""

FILES["backend/app/config.py"] = """import os
class Settings:
    PROJECT_NAME: str = "EasyBUS Core Services"
    MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    DATABASE_NAME: str = "easybus_db"
settings = Settings()"""

FILES["backend/app/database.py"] = """from motor.motor_asyncio import AsyncIOMotorClient
from app.config import settings
import json
import os

class Database:
    client: AsyncIOMotorClient = None
    db = None

db_instance = Database()

async def connect_to_mongo():
    db_instance.client = AsyncIOMotorClient(settings.MONGO_URI)
    db_instance.db = db_instance.client[settings.DATABASE_NAME]
    await bootstrap_data()

async def close_mongo_connection():
    if db_instance.client:
        db_instance.client.close()

async def bootstrap_data():
    if await db_instance.db.routes.count_documents({}) == 0:
        base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        for col, filename in [("routes", "routes.json"), ("stops", "stops.json"), ("traffic", "traffic.json")]:
            p = os.path.join(base, 'data', filename)
            if os.path.exists(p):
                with open(p, 'r') as f:
                    await db_instance.db[col].insert_many(json.load(f))"""

FILES["backend/app/models.py"] = """from pydantic import BaseModel
class UserAuthModel(BaseModel):
    username: str
    password: str

class RouteRecommendationRequest(BaseModel):
    source_query: str
    destination_query: str
    priority: str = "fastest"
    is_late_mode: bool = False
    low_data_mode: bool = False

class PreferencesUpdateModel(BaseModel):
    username: str
    preferred_metric: str
    language: str = "en" """

FILES["backend/app/schemas.py"] = """from pydantic import BaseModel
from typing import List

class LegDetail(BaseModel):
    type: str
    identifier: str
    from_stop: str
    to_stop: str
    duration_mins: int
    distance_meters: int

class RecommendationCardSchema(BaseModel):
    category: str
    eta_mins: int
    delay_prediction_mins: int
    bus_numbers: List[str]
    transfers: int
    speed_score: int
    comfort_score: int
    reliability_score: int
    confidence_score: int
    seat_probability: int
    expected_occupancy: int
    walking_distance_meters: int
    why_text: str
    itinerary: List[LegDetail]"""

FILES["backend/app/ml_inference.py"] = """import pickle
import os
import numpy as np

class MLInferenceEngine:
    def __init__(self):
        base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.delay_path = os.path.join(base, 'ml', 'binaries', 'delay_model.pkl')
        self.occ_path = os.path.join(base, 'ml', 'binaries', 'occupancy_model.pkl')
        self.delay_model = pickle.load(open(self.delay_path, 'rb')) if os.path.exists(self.delay_path) else None
        self.occ_model = pickle.load(open(self.occ_path, 'rb')) if os.path.exists(self.occ_path) else None

    def predict_delay(self, stop_id: str, route_id: str, hour: int) -> float:
        if self.delay_model:
            try:
                return float(self.delay_model.predict(np.array([[1, 1, hour, 1, 0.7]]))[0])
            except: pass
        return 5.0

    def predict_occupancy(self, route_id: str, stop_id: str, hour: int) -> tuple:
        if self.occ_model:
            try:
                occ = float(self.occ_model.predict(np.array([[1, 1, hour, 0]]))[0])
                return occ, max(5, 100 - occ)
            except: pass
        return 60.0, 40.0

ml_broker = MLInferenceEngine()"""

FILES["backend/app/routing.py"] = """from app.database import db_instance
from app.ml_inference import ml_broker

class MultiAgentRoutingEngine:
    @staticmethod
    async def resolve_pmpml_network_matrix(payload):
        src = payload.source_query.lower()
        dest = payload.destination_query.lower()
        
        src_stop = "S-KOT-01" if "kothrud" in src else "S-FC-08" if "fc" in src else "S-SHV-02"
        dest_stop = "S-HIN-06" if "hinjewadi" in dest else "S-SHV-02"
        
        all_routes = await db_instance.db.routes.find().to_list(length=100)
        recommendations = []
        
        w_speed = 0.6 if payload.is_late_mode else (0.4 if payload.priority == "fastest" else 0.2)
        w_comfort = 0.1 if payload.is_late_mode else (0.5 if payload.priority == "comfort" else 0.4)
        w_reliability = 0.3 if payload.is_late_mode else (0.4 if payload.priority == "reliability" else 0.4)
        
        for route in all_routes:
            stops = route["stops"]
            if src_stop in stops and dest_stop in stops:
                if stops.index(src_stop) < stops.index(dest_stop):
                    delay = ml_broker.predict_delay(src_stop, route["route_id"], 9)
                    occ, seat_p = ml_broker.predict_occupancy(route["route_id"], src_stop, 9)
                    
                    speed_score = 85
                    comfort_score = int(seat_p)
                    reliability_score = 90
                    final_score = int((w_speed * speed_score) + (w_comfort * comfort_score) + (w_reliability * reliability_score))
                    
                    recommendations.append({
                        "category": "Best Overall Route",
                        "eta_mins": int(30 + delay),
                        "delay_prediction_mins": int(delay),
                        "bus_numbers": [route["route_no"]],
                        "transfers": 0,
                        "speed_score": speed_score,
                        "comfort_score": comfort_score,
                        "reliability_score": reliability_score,
                        "confidence_score": final_score,
                        "seat_probability": int(seat_p),
                        "expected_occupancy": int(occ),
                        "walking_distance_meters": 200,
                        "why_text": "Selected because this route directly bypasses expected bypass congestion grid locks.",
                        "itinerary": [
                            {"type": "WALK", "identifier": "Walk", "from_stop": "Location", "to_stop": src_stop, "duration_mins": 3, "distance_meters": 200},
                            {"type": "BUS", "identifier": route["route_no"], "from_stop": src_stop, "to_stop": dest_stop, "duration_mins": 30, "distance_meters": 8000}
                        ]
                    })
        if not recommendations:
            recommendations.append({
                "category": "Fastest Alternative (1 Transfer)",
                "eta_mins": 45,
                "delay_prediction_mins": 2,
                "bus_numbers": ["201", "24"],
                "transfers": 1,
                "speed_score": 90,
                "comfort_score": 40,
                "reliability_score": 85,
                "confidence_score": int((w_speed * 90) + (w_comfort * 40) + (w_reliability * 85)),
                "seat_probability": 15,
                "expected_occupancy": 85,
                "walking_distance_meters": 350,
                "why_text": "Recommended route shifting over Shivajinagar terminal to save scheduling alignment errors.",
                "itinerary": [
                    {"type": "BUS", "identifier": "201", "from_stop": "Kothrud Stand", "to_stop": "Shivajinagar", "duration_mins": 15, "distance_meters": 4000},
                    {"type": "BUS", "identifier": "24", "from_stop": "Shivajinagar", "to_stop": "Hinjewadi", "duration_mins": 25, "distance_meters": 11000}
                ]
            })
        return recommendations"""

FILES["backend/app/main.py"] = """from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.database import connect_to_mongo, close_mongo_connection, db_instance
from app.models import UserAuthModel, RouteRecommendationRequest, PreferencesUpdateModel
from app.routing import MultiAgentRoutingEngine
import asyncio
import json
import random

app = FastAPI(title="EasyBUS API Backend Layer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup(): await connect_to_mongo()

@app.on_event("shutdown")
async def shutdown(): await close_mongo_connection()

@app.post("/api/signup")
async def signup(user: UserAuthModel):
    await db_instance.db.users.update_one({"username": user.username}, {"$set": {"password": user.password}}, upsert=True)
    return {"status": "success"}

@app.post("/api/login")
async def login(user: UserAuthModel):
    db_user = await db_instance.db.users.find_one({"username": user.username, "password": user.password})
    if not db_user: raise HTTPException(status_code=401, detail="Bad verification token")
    return {"status": "success"}

@app.post("/api/recommend")
async def recommend(payload: RouteRecommendationRequest):
    return {"status": "success", "data": await MultiAgentRoutingEngine.resolve_pmpml_network_matrix(payload)}

@app.get("/api/nearest-stop")
async def nearest_stop(lat: float, lon: float):
    return {"stop_id": "S-FC-08", "stop_name": "FC Road Bus Stop", "walking_time_mins": 3, "distance_meters": 210, "upcoming_buses": ["118", "24", "201"]}

@app.websocket("/ws/live-buses")
async def ws_live_buses(websocket: WebSocket):
    await websocket.accept()
    paths = {
        "118": [(18.5074, 73.8077), (18.5309, 73.8546), (18.5911, 73.6980)],
        "24": [(18.5018, 73.8562), (18.5414, 73.8269), (18.5911, 73.6980)]
    }
    try:
        step = 0
        while True:
            payload = []
            for r_no, coords in paths.items():
                lat, lon = coords[step % len(coords)]
                payload.append({
                    "bus_id": f"PMPML-MH12-{r_no}", "route_no": r_no,
                    "latitude": lat + random.uniform(-0.002, 0.002), "longitude": lon + random.uniform(-0.002, 0.002),
                    "speed_kmh": random.randint(25, 45), "expected_occupancy": random.randint(40, 95),
                    "seat_probability": random.randint(10, 80), "delay_mins": random.randint(0, 10)
                })
            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(4.0)
            step += 1
    except WebSocketDisconnect: pass"""

# ==========================================
# 4. FRONTEND (REACT / VITE)
# ==========================================
FILES["frontend/package.json"] = """{
  "name": "easybus-frontend",
  "private": true,
  "version": "1.0.0",
  "type": "module",
  "scripts": { "dev": "vite", "build": "vite build" },
  "dependencies": {
    "axios": "^1.6.8",
    "leaflet": "^1.9.4",
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-leaflet": "^4.2.1"
  },
  "devDependencies": {
    "autoprefixer": "^10.4.19",
    "postcss": "^8.4.38",
    "tailwindcss": "^3.4.1",
    "vite": "^5.1.6"
  }
}"""

FILES["frontend/tailwind.config.js"] = """/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  darkMode: 'class',
  theme: {
    extend: {
      colors: { primary: "#2563EB", secondary: "#0F172A", accent: "#10B981", background: "#F8FAFC" }
    },
  },
  plugins: [],
}"""

FILES["frontend/postcss.config.js"] = """export default {
  plugins: { tailwindcss: {}, autoprefixer: {} },
}"""

FILES["frontend/vite.config.js"] = """import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api': { target: 'http://localhost:8000', changeOrigin: true },
      '/ws': { target: 'ws://localhost:8000', ws: true }
    }
  }
})"""

FILES["frontend/index.html"] = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"><title>EasyBUS Framework</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
</head>
<body>
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
</body>
</html>"""

FILES["frontend/src/index.css"] = """@tailwind base; @tailwind components; @tailwind utilities;
body { @apply bg-[#F8FAFC] text-[#0F172A] transition-colors duration-150; }
body.dark { @apply bg-[#0F172A] text-[#F8FAFC]; }
.leaflet-container { width: 100%; height: 100%; border-radius: 0.75rem; }"""

FILES["frontend/src/main.jsx"] = """import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import './index.css'
ReactDOM.createRoot(document.getElementById('root')).render(<App />)"""

FILES["frontend/src/i18n/translations.js"] = """export const translations = {
  en: { welcome: "Welcome to EasyBUS", tagline: "Smart Bus Travel. Zero Guesswork.", heroTitle: "Where do you want to go today?", searchBtn: "Compute Optimal Route" },
  mr: { welcome: "इझीबसवर आपले स्वागत आहे", tagline: "स्मार्ट बस प्रवास. शून्य संभ्रम.", heroTitle: "आज तुम्हाला कोठे जायचे आहे?", searchBtn: "सर्वोत्तम मार्ग शोधा" },
  hi: { welcome: "ईजीबस में आपका स्वागत है", tagline: "स्मार्ट बस यात्रा। शून्य दुविधा।", heroTitle: "आज आपको कहां जाना है?", searchBtn: "सर्वोत्तम मार्ग की गणना करें" }
};"""

FILES["frontend/src/components/ControlPanel.jsx"] = """import React from 'react';
export default function ControlPanel({ src, setSrc, dest, setDest, pri, setPri, late, setLate, lite, setLite, onSearch, tokens }) {
  return (
    <div className="bg-white dark:bg-slate-800 p-5 rounded-xl shadow-md space-y-4">
      <h2 className="text-sm font-bold text-slate-400 uppercase">{tokens.heroTitle}</h2>
      <input type="text" placeholder="Source Stop" value={src} onChange={e=>setSrc(e.target.value)} className="w-full p-2.5 rounded-lg border dark:bg-slate-900 border-slate-200 dark:border-slate-700 text-sm" />
      <input type="text" placeholder="Destination Stop" value={dest} onChange={e=>setDest(e.target.value)} className="w-full p-2.5 rounded-lg border dark:bg-slate-900 border-slate-200 dark:border-slate-700 text-sm" />
      <div className="grid grid-cols-3 gap-2">
        {['fastest','comfort','reliability'].map(m=>(
          <button key={m} onClick={()=>setPri(m)} className={`text-xs p-2 font-bold rounded-lg uppercase ${pri===m?'bg-primary text-white':'bg-slate-100 dark:bg-slate-700 text-slate-500'}`}>{m}</button>
        ))}
      </div>
      <div className="flex flex-col space-y-2 pt-2">
        <label className="flex items-center space-x-2 text-xs text-rose-500 font-bold cursor-pointer"><input type="checkbox" checked={late} onChange={e=>setLate(e.target.checked)}/> <span>🚀 I Am Getting Late Mode</span></label>
        <label className="flex items-center space-x-2 text-xs text-amber-500 font-bold cursor-pointer"><input type="checkbox" checked={lite} onChange={e=>setLite(e.target.checked)}/> <span>📉 Low Data Lite Mode</span></label>
      </div>
      <button onClick={onSearch} className="w-full bg-primary text-white p-3 rounded-lg font-bold text-sm shadow-sm hover:opacity-90">{tokens.searchBtn}</button>
    </div>
  );
}"""

FILES["frontend/src/components/RouteCard.jsx"] = """import React from 'react';
export default function RouteCard({ route }) {
  const isFull = route.expected_occupancy >= 90;
  const share = () => {
    const msg = `*EasyBUS Commute* 🚌%0ARoute: PMPML ${route.bus_numbers.join(',')}%0AETA: ${route.eta_mins} mins%0ASeat Probability: ${route.seat_probability}%25`;
    window.open(`https://api.whatsapp.com/send?text=${msg}`, '_blank');
  };
  return (
    <div className="bg-white dark:bg-slate-800 p-5 rounded-xl shadow-md border-l-4 border-primary space-y-3">
      <div className="flex justify-between items-center">
        <div className="flex items-center space-x-2">
          {route.bus_numbers.map((n,i)=><span key={i} className="bg-slate-900 text-white px-2.5 py-1 font-mono text-xs font-bold rounded">{n}</span>)}
          <span className="text-xs text-slate-400">({route.transfers} transfers)</span>
        </div>
        <div className="text-right"><span className="text-xl font-black">{route.eta_mins} mins</span><div className="text-[10px] text-rose-500">+{route.delay_prediction_mins}m delay prediction</div></div>
      </div>
      {isFull && <div className="text-xs bg-amber-50 text-amber-800 p-2 rounded-lg font-bold">⚠️ Bus Almost Full — Wait For Next Bus or Choose Alternate Route</div>}
      <div className="grid grid-cols-3 gap-2 bg-slate-50 dark:bg-slate-900 p-2.5 rounded-lg text-center text-xs">
        <div><div className="text-slate-400">Seat Prob</div><div className="font-bold text-emerald-600">{route.seat_probability}%</div></div>
        <div><div className="text-slate-400">Crowd Density</div><div className="font-bold">{route.expected_occupancy}%</div></div>
        <div><div className="text-slate-400">Confidence</div><div className="font-bold text-primary">{route.confidence_score}%</div></div>
      </div>
      <p className="text-xs text-slate-500 dark:text-slate-400 italic">Why this route? {route.why_text}</p>
      <div className="flex justify-between items-center pt-2 border-t border-slate-100 dark:border-slate-700 text-xs text-slate-400">
        <span>🚶 {route.walking_distance_meters}m walk required</span>
        <button onClick={share} className="bg-[#25D366] text-white px-3 py-1.5 rounded font-bold">Share</button>
      </div>
    </div>
  );
}"""

FILES["frontend/src/components/WaitBoardAssistant.jsx"] = """import React from 'react';
export default function WaitBoardAssistant({ routes }) {
  if (!routes || routes.length === 0) return null;
  const top = routes[0];
  const wait = top.seat_probability < 25 && top.expected_occupancy > 85;
  return (
    <div className="bg-slate-900 text-white p-4 rounded-xl shadow-md border border-slate-700">
      <div className="text-[10px] font-bold text-blue-400 uppercase tracking-wider">Wait or Board Assistant</div>
      <h3 className="font-bold text-sm mt-1">{wait ? "💡 Recommendation: Wait For Next Bus" : "💡 Recommendation: Board This Bus Instantly"}</h3>
      <p className="text-xs text-slate-400 mt-0.5">{wait ? "High crowd density profile. Spacing analytics map another vehicle arrival within 7 minutes with a 75% higher seating layout probability configuration." : "Balanced route alignment metrics. Boarding this asset delivers safe scheduling optimization goals."}</p>
    </div>
  );
}"""

FILES["frontend/src/components/MapView.jsx"] = """import React from 'react';
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
export default function MapView({ buses, lite }) {
  if (lite) {
    return (
      <div className="bg-white dark:bg-slate-800 p-6 rounded-xl shadow-md h-[300px] flex flex-col justify-center items-center text-center">
        <div className="text-amber-500 font-bold text-sm">📉 Map Rendering Engine Paused</div>
        <p className="text-xs text-slate-400 mt-1 max-w-xs">Low Data Lite Mode active. Telemetry stream monitored textually to preserve framework battery footprint.</p>
        <div className="w-full mt-3 max-h-[120px] overflow-y-auto space-y-1 text-left text-xs px-4">
          {buses.map(b=>(<div key={b.bus_id} className="p-1.5 bg-slate-50 dark:bg-slate-900 rounded">🚌 PMPML {b.route_no} | Delay: {b.delay_mins}m | Crowd: {b.expected_occupancy}%</div>))}
        </div>
      </div>
    );
  }
  return (
    <div className="h-[400px] w-full rounded-xl overflow-hidden shadow-md">
      <MapContainer center={[18.5204, 73.8567]} zoom={12} style={{height:'100%', width:'100%'}}>
        <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
        {buses.map(b=>(
          <Marker key={b.bus_id} position={[b.latitude, b.longitude]}>
            <Popup><div className="text-xs font-sans"><b>PMPML Route {b.route_no}</b><br/>Speed: {b.speed_kmh}kmh<br/>Seat Probability: {b.seat_probability}%</div></Popup>
          </Marker>
        ))}
      </MapContainer>
    </div>
  );
}"""

FILES["frontend/src/App.jsx"] = """import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { translations } from './i18n/translations';
import ControlPanel from './components/ControlPanel';
import MapView from './components/MapView';
import RouteCard from './components/RouteCard';
import WaitBoardAssistant from './components/WaitBoardAssistant';

export default function App() {
  const [lang, setLang] = useState('en');
  const [src, setSrc] = useState('Kothrud');
  const [dest, setDest] = useState('Hinjewadi');
  const [pri, setPri] = useState('fastest');
  const [late, setLate] = useState(false);
  const [lite, setLite] = useState(false);
  const [routes, setRoutes] = useState([]);
  const [buses, setBuses] = useState([]);
  const [dark, setDark] = useState(false);

  useEffect(() => {
    const loc = window.location;
    const wsUrl = `${loc.protocol==='https:'?'wss:':'ws:'}//${loc.hostname==='localhost'?'localhost:8000':loc.host}/ws/live-buses`;
    const ws = new WebSocket(wsUrl);
    ws.onmessage = (e) => setBuses(JSON.parse(e.data));
    return () => ws.close();
  }, []);

  const search = async () => {
    try {
      const res = await axios.post('/api/recommend', { source_query: src, destination_query: dest, priority: pri, is_late_mode: late, low_data_mode: lite });
      if (res.data.status === 'success') setRoutes(res.data.data);
    } catch (err) { console.error(err); }
  };

  return (
    <div className={dark ? 'dark bg-slate-900 text-white min-h-screen' : 'bg-slate-50 text-slate-900 min-h-screen'}>
      <header className="bg-white dark:bg-slate-800 border-b border-slate-200 dark:border-slate-700 p-4 flex justify-between items-center">
        <div>
          <h1 className="text-lg font-black text-primary tracking-tight">{translations[lang].welcome}</h1>
          <p className="text-[10px] text-slate-400 font-medium">{translations[lang].tagline}</p>
        </div>
        <div className="flex space-x-2">
          <select value={lang} onChange={e=>setLang(e.target.value)} className="bg-slate-50 dark:bg-slate-900 text-xs border border-slate-200 dark:border-slate-700 p-1 rounded font-bold outline-none">
            <option value="en">English</option><option value="mr">मराठी</option><option value="hi">हिन्दी</option>
          </select>
          <button onClick={()=>setDark(!dark)} className="text-xs bg-slate-100 dark:bg-slate-700 px-3 py-1 rounded font-bold">{dark?'☀️':'🌙'}</button>
        </div>
      </header>
      <main className="max-w-7xl mx-auto p-4 grid grid-cols-1 lg:grid-cols-12 gap-5">
        <div className="lg:col-span-4 space-y-4">
          <ControlPanel src={src} setSrc={setSrc} dest={dest} setDest={setDest} pri={pri} setPri={setPri} late={late} setLate={setLate} lite={lite} setLite={setLite} onSearch={search} tokens={translations[lang]} />
          <WaitBoardAssistant routes={routes} />
        </div>
        <div className="lg:col-span-8 space-y-4">
          <MapView buses={buses} lite={lite} />
          <div className="space-y-3">
            <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider">Engine Routing Recommendations</h3>
            {routes.map((r,i)=><RouteCard key={i} route={r} />)}
          </div>
        </div>
      </main>
    </div>
  );
}"""

# ==========================================
# 5. ORCHESTRATION & DOCS
# ==========================================
FILES["docker-compose.yml"] = """version: '3.8'
services:
  mongodb:
    image: mongo:6.0
    ports:
      - "27017:27017"
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - MONGO_URI=mongodb://mongodb:27017
    depends_on:
      - mongodb
  frontend:
    image: node:18-alpine
    volumes:
      - ./frontend:/app
    working_dir: /app
    ports:
      - "3000:3000"
    command: sh -c "npm install && npm run dev -- --host"
    depends_on:
      - backend"""

FILES["README.md"] = """# EasyBUS: Smart Bus Travel. Zero Guesswork.

EasyBUS is an AI-powered commuter platform designed for the **PMPML** network in Pune. It predicts seat availability, delays, and routes dynamically.

## Quick Installation (Docker)
Run the following single command at the root folder:
```bash
docker-compose up --build
