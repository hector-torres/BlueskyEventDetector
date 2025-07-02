"""
Main logic for detecting event clusters, scoring, timeliness, and storing cluster-level results.
Clusters are the default unit of measure, including single-post clusters.
Logging summarizes run start and final summary of clusters and posts saved.
"""
import os
import re
import logging
import uuid
import warnings
from datetime import datetime, timedelta
import math
import spacy
from spacy.cli import download as spacy_download
from typing import List, Dict
from app.database import get_posts_connection, get_event_connection

# Suppress spaCy word-vector warnings
warnings.filterwarnings("ignore", message=r".*has no word vectors loaded.*")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
DEFAULT_SPACY_MODEL = os.getenv("SPACY_MODEL_NAME", "en_core_web_md")
FALLBACK_SPACY_MODEL = "en_core_web_sm"
CLUSTER_SIMILARITY_THRESHOLD = float(os.getenv("CLUSTER_SIMILARITY_THRESHOLD", "0.8"))

# Load spaCy model with fallback
def load_spacy_model(model_name: str = DEFAULT_SPACY_MODEL):
    try:
        logger.info(f"Loading spaCy model '{model_name}' for similarity")
        return spacy.load(model_name)
    except OSError:
        logger.info(f"Model '{model_name}' not found, downloading...")
        try:
            spacy_download(model_name)
            return spacy.load(model_name)
        except Exception:
            logger.warning(f"Failed to load '{model_name}', falling back to '{FALLBACK_SPACY_MODEL}'")
    try:
        spacy_download(FALLBACK_SPACY_MODEL)
    except Exception:
        pass
    return spacy.load(FALLBACK_SPACY_MODEL)

nlp = load_spacy_model()

# Heuristic keyword lists
OPINION_KEYWORDS = [r"\bOpinion\b", r"\bEditorial\b", r"\bOp-Ed\b", r"\bAnalysis\b"]
PROMO_KEYWORDS   = [r"\bSubscribe\b", r"\bRegister\b", r"\bJoin us\b", r"\bSponsored\b", r"\bSign up\b", r"\bLearn more\b"]

def contains_keyword(text: str, patterns: List[str]) -> bool:
    if not text:
        return False
    text_lower = text.lower()
    return any(re.search(pat.lower(), text_lower) for pat in patterns)

def is_non_event_opinion_or_promo(record) -> bool:
    for field in ("title", "text", "page_title", "meta_description"):
        if contains_keyword(record[field], OPINION_KEYWORDS) or contains_keyword(record[field], PROMO_KEYWORDS):
            return True
    return False

def has_event_entities(text: str) -> bool:
    doc = nlp(text or "")
    has_date = any(ent.label_ == "DATE" for ent in doc.ents)
    has_entity = any(ent.label_ in ("PERSON", "ORG", "GPE", "LOC") for ent in doc.ents)
    return has_date and has_entity

def predict_event_probability(text: str) -> float:
    # TODO: integrate a trained classification model for soft scores
    return 1.0 if has_event_entities(text) else 0.0

def compute_event_annotations() -> List[Dict]:
    """Fetch posts and annotate each with newsworthiness_score."""
    conn = get_posts_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT uuid, title, text, timestamp, page_title, meta_description FROM posts"
    )
    rows = cursor.fetchall()
    conn.close()

    events = []
    for row in rows:
        combined = " ".join(filter(None, [
            row["title"], row["text"], row["page_title"], row["meta_description"]
        ]))
        score = 0.0 if is_non_event_opinion_or_promo(row) else predict_event_probability(combined)
        events.append({
            "post_id": str(row["uuid"]),
            "timestamp": row["timestamp"],
            "newsworthiness_score": score,
            "combined_text": combined
        })
    return events

def cluster_events(events: List[Dict]) -> None:
    """Annotate events with cluster IDs and average newsworthiness scores."""
    now = datetime.utcnow()
    recent = []
    for e in events:
        ts_str = e["timestamp"]
        if not ts_str:
            continue
        try:
            ts = datetime.fromisoformat(ts_str.rstrip("Z"))
        except ValueError:
            continue
        if ts >= now - timedelta(hours=24) and e["newsworthiness_score"] > 0.0:
            recent.append(e)

    docs = {e["post_id"]: nlp(e["combined_text"]) for e in recent}
    parent = {pid: pid for pid in docs}
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    def union(u, v):
        pu, pv = find(u), find(v)
        if pu != pv:
            parent[pv] = pu

    ids = list(docs)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if docs[ids[i]].similarity(docs[ids[j]]) >= CLUSTER_SIMILARITY_THRESHOLD:
                union(ids[i], ids[j])

    clusters = {}
    for pid in ids:
        root = find(pid)
        clusters.setdefault(root, []).append(pid)

    cluster_map: Dict[str, tuple] = {}
    for members in clusters.values():
        cid = str(uuid.uuid4())
        member_scores = [e["newsworthiness_score"] for e in events if e["post_id"] in members]
        avg_score = sum(member_scores) / len(member_scores) if member_scores else 0.0
        for pid in members:
            cluster_map[pid] = (cid, avg_score)

    for e in events:
        if e["post_id"] not in cluster_map:
            single_id = str(uuid.uuid4())
            cluster_map[e["post_id"]] = (single_id, e["newsworthiness_score"])

    for e in events:
        e["cluster_id"], e["cluster_prob"] = cluster_map[e["post_id"]]


def detect_clusters() -> List[Dict]:
    """Compute clusters from annotated events, including timeliness."""
    events = compute_event_annotations()
    cluster_events(events)
    now = datetime.utcnow()
    clusters: Dict[str, Dict] = {}
    for e in events:
        cid = e["cluster_id"]
        clusters.setdefault(cid, {
            "cluster_id": cid,
            "cluster_prob": e["cluster_prob"],
            "post_ids": [],
            "timestamps": []
        })
        clusters[cid]["post_ids"].append(e["post_id"])
        clusters[cid]["timestamps"].append(e["timestamp"])

    cluster_list = []
    for c in clusters.values():
        times = []
        for ts_str in c["timestamps"]:
            try:
                times.append(datetime.fromisoformat(ts_str.rstrip("Z")))
            except Exception:
                pass
        first = min(times) if times else now
        diff_minutes = (now - first).total_seconds() / 60
        if diff_minutes <= 5:
            timeliness = 1.0
        else:
            intervals = math.floor((diff_minutes - 5) / 30)
            timeliness = max(0.0, 1.0 - 0.05 * intervals)
        c["timeliness"] = timeliness
        cluster_list.append(c)
    return cluster_list

def write_clusters_to_db(clusters: List[Dict]) -> None:
    """Write cluster-level records to the event database, including newsworthiness and timeliness."""
    conn = get_event_connection()
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS clusters")
    cur.execute(
        """
        CREATE TABLE clusters (
            cluster_id TEXT PRIMARY KEY,
            newsworthiness_score REAL,
            timeliness REAL,
            post_ids TEXT
        )
        """
    )
    for c in clusters:
        post_ids_str = ",".join(c["post_ids"])
        cur.execute(
            "INSERT OR REPLACE INTO clusters (cluster_id, newsworthiness_score, timeliness, post_ids) VALUES (?, ?, ?, ?)",
            (c["cluster_id"], c["cluster_prob"], c.get("timeliness"), post_ids_str)
        )
    conn.commit()
    conn.close()

def run():
    logger.info("Event clustering run started")
    clusters = detect_clusters()
    write_clusters_to_db(clusters)
    total_posts = sum(len(c["post_ids"]) for c in clusters)
    logger.info(f"Run completed: {len(clusters)} clusters (created from {total_posts} posts) saved to database")
