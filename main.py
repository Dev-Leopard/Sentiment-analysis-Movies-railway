from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
import requests
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# TMDb and Sentiment model setup
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE_URL = "https://api.themoviedb.org/3"
sentiment_model = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/search", response_class=HTMLResponse)
async def search_movie(request: Request, movie: str = Form(...)):
    response = requests.get(f"{TMDB_BASE_URL}/search/movie", params={
        "api_key": TMDB_API_KEY,
        "query": movie
    })
    results = response.json().get("results", [])[:3]
    movies = [{
        "title": r.get("title"),
        "id": r.get("id"),
        "year": r.get("release_date", "N/A")[:4]
    } for r in results]
    return templates.TemplateResponse("index.html", {"request": request, "movies": movies})

@app.get("/movie/{movie_id}", response_class=HTMLResponse)
async def movie_detail(request: Request, movie_id: int):
    try:
        # Fetch metadata
        movie = requests.get(f"{TMDB_BASE_URL}/movie/{movie_id}", params={
            "api_key": TMDB_API_KEY,
            "language": "en-US"
        }).json()

        credits = requests.get(f"{TMDB_BASE_URL}/movie/{movie_id}/credits", params={
            "api_key": TMDB_API_KEY
        }).json()

        reviews_raw = []
        for page in range(1, 6):  # Fetch up to 100 reviews
            r = requests.get(f"{TMDB_BASE_URL}/movie/{movie_id}/reviews", params={
                "api_key": TMDB_API_KEY,
                "language": "en-US",
                "page": page
            }).json().get("results", [])
            if not r:
                break
            reviews_raw.extend(r)

        texts = [r["content"][:500] for r in reviews_raw[:100] if "content" in r]
        sentiments = sentiment_model(texts, batch_size=8)

        analyzed = []
        pos_count = 0
        for i, s in enumerate(sentiments):
            label = "positive" if s["label"].startswith(("4", "5")) else "negative"
            if label == "positive":
                pos_count += 1
            analyzed.append({
                "content": reviews_raw[i].get("content", ""),
                "author": reviews_raw[i].get("author", "Anonymous"),
                "rating": reviews_raw[i].get("author_details", {}).get("rating", "N/A"),
                "sentiment": s["label"],
                "label": label
            })

        total_reviews = len(analyzed)
        pos_percent = round((pos_count / total_reviews) * 100) if total_reviews else 0
        neg_percent = 100 - pos_percent

        # Movie metadata for display
        poster = f"https://image.tmdb.org/t/p/w500{movie.get('poster_path')}" if movie.get('poster_path') else ""
        genres = ', '.join([g["name"] for g in movie.get("genres", [])])
        director = next((m["name"] for m in credits.get("crew", []) if m["job"] == "Director"), "Unknown")
        cast = ', '.join([a["name"] for a in credits.get("cast", [])[:3]])

        return templates.TemplateResponse("index.html", {
            "request": request,
            "title": movie.get("title"),
            "poster": poster,
            "rating": movie.get("vote_average"),
            "runtime": movie.get("runtime"),
            "genres": genres,
            "director": director,
            "overview": movie.get("overview"),
            "cast": cast,
            "reviews": analyzed,
            "positive_percent": pos_percent,
            "negative_percent": neg_percent,
            "positive_count": pos_count,
            "negative_count": total_reviews - pos_count
        })

    except Exception as e:
        print("Error:", e)
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": "Failed to fetch movie data or reviews."
        })
