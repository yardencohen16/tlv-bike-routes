"""
server.py — FastAPI server wrapping the flat-route cycling engine.

Startup loads the graph once into memory (cached to disk after first run).

Run locally:
    python3 server.py

Endpoint:
    GET /route?start_lat=32.0853&start_lon=34.7818&end_lat=32.0679&end_lon=34.7751
"""

import os
import pickle
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
import uvicorn

from flat_route import load_graph, fetch_elevations, annotate_elevation_gain, get_flat_route, get_shortest_route, compute_route_stats

CACHE_FILE = "graph_cache.pkl"

# ---------------------------------------------------------------------------
# Graph — loaded once at startup, cached to disk
# ---------------------------------------------------------------------------

state: dict = {}


def load_or_build_graph():
    if os.path.exists(CACHE_FILE):
        print("Loading graph from cache (fast)...")
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)

    print("First run — downloading graph and fetching elevations (this takes a few minutes)...")
    G = load_graph()
    G = fetch_elevations(G)
    G = annotate_elevation_gain(G)

    print("Saving graph to cache for future restarts...")
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(G, f)

    return G


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["graph"] = load_or_build_graph()
    print("Graph ready. Server is up.")
    yield
    state.clear()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

limiter = Limiter(key_func=get_remote_address, default_limits=["10/minute"])

app = FastAPI(title="TLV Flat Bike Router", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda req, exc: __import__('fastapi').responses.JSONResponse(
    status_code=429, content={"detail": "Too many requests — please wait a moment."}
))
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Serve the UI + static assets
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.get("/logo.png")
def logo():
    return FileResponse(os.path.join(BASE_DIR, "logo.png"))


# ---------------------------------------------------------------------------
# Route endpoint
# ---------------------------------------------------------------------------

@app.get("/route")
@limiter.limit("10/minute")
def route(
    request: Request,
    start_lat: float = Query(..., description="Start latitude"),
    start_lon: float = Query(..., description="Start longitude"),
    end_lat:   float = Query(..., description="End latitude"),
    end_lon:   float = Query(..., description="End longitude"),
):
    G = state.get("graph")
    if G is None:
        raise HTTPException(status_code=503, detail="Graph not loaded yet")

    try:
        coords = get_flat_route(G, (start_lat, start_lon), (end_lat, end_lon))
        distance_km, elevation_gain_m = compute_route_stats(G, coords)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "distance_km": round(distance_km, 3),
        "elevation_gain_m": round(elevation_gain_m, 1),
        "waypoints": len(coords),
        "coordinates": [{"lat": lat, "lon": lon} for lat, lon in coords],
    }


# ---------------------------------------------------------------------------
# Compare endpoint — returns both flat and shortest routes
# ---------------------------------------------------------------------------

@app.get("/compare")
@limiter.limit("10/minute")
def compare(
    request: Request,
    start_lat: float = Query(...),
    start_lon: float = Query(...),
    end_lat:   float = Query(...),
    end_lon:   float = Query(...),
):
    G = state.get("graph")
    if G is None:
        raise HTTPException(status_code=503, detail="Graph not loaded yet")

    try:
        origin = (start_lat, start_lon)
        dest   = (end_lat,   end_lon)

        flat_coords    = get_flat_route(G, origin, dest)
        short_coords   = get_shortest_route(G, origin, dest)

        flat_dist,  flat_gain  = compute_route_stats(G, flat_coords)
        short_dist, short_gain = compute_route_stats(G, short_coords)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "flat": {
            "distance_km":      round(flat_dist,  3),
            "elevation_gain_m": round(flat_gain,  1),
            "waypoints":        len(flat_coords),
            "coordinates":      [{"lat": lat, "lon": lon} for lat, lon in flat_coords],
        },
        "shortest": {
            "distance_km":      round(short_dist,  3),
            "elevation_gain_m": round(short_gain,  1),
            "waypoints":        len(short_coords),
            "coordinates":      [{"lat": lat, "lon": lon} for lat, lon in short_coords],
        },
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
