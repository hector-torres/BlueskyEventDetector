import os
import sys

# Ensure project root is in PYTHONPATH for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import streamlit as st
from app.database import get_posts_connection, get_event_connection

# Constants
DISPLAY_ROWS = 10

# Data loading functions
def load_clusters():
    conn = get_event_connection()
    df = pd.read_sql_query(
        "SELECT cluster_id, newsworthiness_score, timeliness, post_ids FROM clusters",
        conn
    )
    conn.close()
    return df


def load_posts(post_ids):
    conn = get_posts_connection()
    rows = []
    for pid in post_ids:
        row = conn.execute(
            "SELECT author, title, text, timestamp FROM posts WHERE uuid = ?",
            (pid,)
        ).fetchone()
        if row:
            rows.append(dict(row))
    conn.close()
    return pd.DataFrame(rows)


def get_latest_post_info(post_ids_str):
    if not post_ids_str:
        return None, None, None, None
    ids = post_ids_str.split(',')
    posts_df = load_posts(ids)
    if posts_df.empty:
        return None, None, None, None
    posts_df['timestamp_dt'] = pd.to_datetime(posts_df['timestamp'], utc=True)
    latest = posts_df.sort_values('timestamp_dt', ascending=False).iloc[0]
    return (
        latest.get('author'),
        latest.get('title'),
        latest.get('text'),
        latest.get('timestamp')
    )

# Main UI
st.title("Event Clusters Dashboard")
clusters_df = load_clusters()

if clusters_df.empty:
    st.info("No clusters found in the database.")
else:
    # Compute post_count and latest post info
    clusters_df['post_count'] = clusters_df['post_ids'].apply(lambda x: len(x.split(',')) if x else 0)
    latest_info = clusters_df['post_ids'].apply(get_latest_post_info)
    clusters_df[['latest_author', 'latest_title', 'latest_text', 'latest_timestamp']] = \
        pd.DataFrame(latest_info.tolist(), index=clusters_df.index)

    # Sort by newest timestamp and limit rows
    clusters_df['latest_timestamp_dt'] = pd.to_datetime(clusters_df['latest_timestamp'], utc=True)
    sorted_df = clusters_df.sort_values('latest_timestamp_dt', ascending=False).head(DISPLAY_ROWS)

    # Prepare display dataframe with reordered columns
    display_df = sorted_df[[
        'latest_author',
        'latest_title',
        'latest_text',
        'latest_timestamp',
        'newsworthiness_score',
        'timeliness',
        'post_count'
    ]].copy()
    # Rename columns: underscores -> spaces, title case
    display_df.columns = [col.replace('_', ' ').title() for col in display_df.columns]

    # Add clickable View Posts link
    sorted_df['view_posts'] = sorted_df['cluster_id'].apply(
        lambda cid: f'<a href="?cluster={cid}">View Posts</a>'
    )
    display_df['View Posts'] = sorted_df['view_posts'].values

    # Render table as HTML
    st.write(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)
