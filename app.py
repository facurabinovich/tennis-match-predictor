"""
Tennis Match Predictor - Database Version
Production model: LightGBM GridSearch (84 features, 67.67% test accuracy)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from config import get_db_connection

# ============================================================
# PAGE CONFIG — must be first Streamlit call
# ============================================================

st.set_page_config(
    page_title="Tennis Match Predictor",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# AUTO DATA UPDATE (once per session)
# ============================================================

if 'data_updated' not in st.session_state:
    with st.spinner("🔄 Checking for new match data..."):
        try:
            from data_updater import run_update
            updated = run_update(year=datetime.now().year)
            if updated:
                st.success(f"✅ Database updated with latest {datetime.now().year} matches!")
                st.cache_data.clear()
            else:
                st.info("✓ Database is up to date")
            st.session_state.data_updated = True
        except Exception as e:
            st.warning(f"⚠️ Could not check for updates: {e}")
            st.session_state.data_updated = True

if 'cache_cleared' not in st.session_state:
    st.cache_data.clear()
    st.cache_resource.clear()
    st.session_state.cache_cleared = True

# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():
    import pickle
    with open('models/lgbm_final.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/feature_set.pkl', 'rb') as f:
        feature_config = pickle.load(f)
    with open('models/model_metrics.pkl', 'rb') as f:
        metrics = pickle.load(f)
    return {
        'model':    model,
        'features': feature_config['features'],
        'metrics':  metrics,
        'params':   feature_config.get('best_params', {})
    }

model_data = load_model()

# ============================================================
# DATABASE QUERY FUNCTIONS (all top-level for proper caching)
# ============================================================

@st.cache_data(ttl=3600)
def get_players_list():
    connection = get_db_connection()
    if not connection:
        return []
    cursor = connection.cursor()
    cursor.execute("""
        SELECT DISTINCT p.player_name
        FROM players p
        JOIN matches m ON (p.player_id = m.winner_id OR p.player_id = m.loser_id)
        WHERE m.date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
        ORDER BY p.player_name
    """)
    players = [row[0] for row in cursor.fetchall()]
    connection.close()
    return players

@st.cache_data(ttl=3600)
def get_player_stats(player_name):
    connection = get_db_connection()
    if not connection:
        return None
    cursor = connection.cursor(dictionary=True)
    cursor.execute("""
        SELECT
            ps.current_rank, ps.current_rank_points, TIMESTAMPDIFF(YEAR, p.birth_date, CURDATE()) AS age, ps.height,
            ps.elo_overall, ps.elo_hard, ps.elo_clay, ps.elo_grass, ps.elo_carpet,
            ps.wins_last_10, ps.matches_last_10, ps.matches_last_15d,
            ps.last_match_date, ps.days_since_last_match,
            ps.form_surface_20,
            ps.form_level_G, ps.form_level_M, ps.form_level_500, ps.form_level_250,
            ps.first_serve_pct, ps.first_serve_won_pct, ps.second_serve_won_pct,
            ps.bp_save_pct, ps.ace_rate, ps.df_rate,
            ps.first_serve_return_won_pct, ps.second_serve_return_won_pct,
            ps.bp_conversion_pct,
            p.nationality, p.hand
        FROM player_stats ps
        JOIN players p ON ps.player_id = p.player_id
        WHERE p.player_name = %s
    """, (player_name,))
    stats = cursor.fetchone()
    connection.close()
    return stats

@st.cache_data(ttl=600)
def get_recent_matches(player_name, limit=5):
    connection = get_db_connection()
    if not connection:
        return []
    cursor = connection.cursor(dictionary=True)
    cursor.execute("""
        SELECT m.date, m.tournament, m.surface, m.round,
               m.winner_name, m.loser_name, m.score, m.minutes,
               CASE WHEN m.winner_name = %s THEN 'W' ELSE 'L' END as result
        FROM matches m
        WHERE m.winner_name = %s OR m.loser_name = %s
        ORDER BY m.date DESC
        LIMIT %s
    """, (player_name, player_name, player_name, limit))
    matches = cursor.fetchall()
    connection.close()
    return matches

@st.cache_data(ttl=600)
def get_h2h_matches(player_a_name, player_b_name):
    connection = get_db_connection()
    if not connection:
        return []
    cursor = connection.cursor(dictionary=True)
    cursor.execute("""
        SELECT m.date, m.tournament, m.surface, m.round,
               m.winner_name, m.loser_name, m.score
        FROM matches m
        WHERE (m.winner_name = %s AND m.loser_name = %s)
           OR (m.winner_name = %s AND m.loser_name = %s)
        ORDER BY m.date DESC
    """, (player_a_name, player_b_name, player_b_name, player_a_name))
    matches = cursor.fetchall()
    connection.close()
    return matches

@st.cache_data(ttl=600)
def get_h2h(player_a_name, player_b_name):
    matches = get_h2h_matches(player_a_name, player_b_name)
    if not matches:
        return None
    
    total = len(matches)
    a_wins = sum(1 for m in matches if m['winner_name'] == player_a_name)
    b_wins = total - a_wins
    
    last_3 = matches[:3]  # already sorted DESC
    a_wins_last_3 = sum(1 for m in last_3 if m['winner_name'] == player_a_name)
    b_wins_last_3 = len(last_3) - a_wins_last_3
    
    return {
        'player_a_name': player_a_name,
        'player_b_name': player_b_name,
        'total_matches': total,
        'player_a_wins': a_wins,
        'player_b_wins': b_wins,
        'last_3_matches': len(last_3),
        'player_a_wins_last_3': a_wins_last_3,
        'player_b_wins_last_3': b_wins_last_3,
    }

@st.cache_data(ttl=600)
def get_season_record(player_name, year):
    connection = get_db_connection()
    if not connection:
        return 0, 0
    cursor = connection.cursor()
    cursor.execute("""
        SELECT SUM(CASE WHEN winner_name = %s THEN 1 ELSE 0 END),
               SUM(CASE WHEN loser_name  = %s THEN 1 ELSE 0 END)
        FROM matches
        WHERE YEAR(date) = %s AND (winner_name = %s OR loser_name = %s)
    """, (player_name, player_name, year, player_name, player_name))
    result = cursor.fetchone()
    connection.close()
    return result[0] or 0, result[1] or 0

@st.cache_data(ttl=600)
def get_days_since_last_match(player_name):
    connection = get_db_connection()
    if not connection:
        return 999
    cursor = connection.cursor()
    cursor.execute("""
        SELECT MAX(date) FROM matches
        WHERE winner_name = %s OR loser_name = %s
    """, (player_name, player_name))
    result = cursor.fetchone()
    connection.close()
    if result and result[0]:
        return (datetime.now().date() - result[0]).days
    return 999

@st.cache_data(ttl=600)
def get_tournament_history(player_name, tournament_name):
    connection = get_db_connection()
    if not connection:
        return None
    cursor = connection.cursor(dictionary=True)
    cursor.execute("""
        SELECT m.date, m.round, m.winner_name, m.loser_name, m.score,
               CASE WHEN m.winner_name = %s THEN 'W' ELSE 'L' END as result
        FROM matches m
        WHERE (m.winner_name = %s OR m.loser_name = %s) AND m.tournament = %s
        ORDER BY m.date DESC LIMIT 1
    """, (player_name, player_name, player_name, tournament_name))
    result = cursor.fetchone()
    connection.close()
    return result

@st.cache_data(ttl=600)
def get_surface_record(player_name, surface_name, year):
    connection = get_db_connection()
    if not connection:
        return 0, 0
    cursor = connection.cursor()
    cursor.execute("""
        SELECT SUM(CASE WHEN winner_name = %s THEN 1 ELSE 0 END),
               SUM(CASE WHEN loser_name  = %s THEN 1 ELSE 0 END)
        FROM matches
        WHERE surface = %s AND YEAR(date) = %s
          AND (winner_name = %s OR loser_name = %s)
    """, (player_name, player_name, surface_name, year, player_name, player_name))
    r = cursor.fetchone()
    connection.close()
    return r[0] or 0, r[1] or 0

# ============================================================
# SEASON DASHBOARD QUERIES
# ============================================================

@st.cache_data(ttl=3600)
def get_season_overview(year):
    connection = get_db_connection()
    if not connection:
        return {}
    cursor = connection.cursor(dictionary=True)
    cursor.execute("""
        SELECT COUNT(*) as total_matches,
               COUNT(DISTINCT tournament) as total_tournaments,
               AVG(minutes) as avg_duration
        FROM matches WHERE YEAR(date) = %s
    """, (year,))
    overview = cursor.fetchone()
    cursor.execute("""
        SELECT COUNT(DISTINCT winner_name) + COUNT(DISTINCT loser_name) as total_players
        FROM matches WHERE YEAR(date) = %s
    """, (year,))
    overview['total_players'] = cursor.fetchone()['total_players']
    cursor.execute("""
        SELECT surface, COUNT(*) as matches FROM matches
        WHERE YEAR(date) = %s AND surface IS NOT NULL
        GROUP BY surface ORDER BY matches DESC
    """, (year,))
    overview['surfaces'] = cursor.fetchall()
    cursor.execute("""
        SELECT tourney_level, COUNT(*) as matches FROM matches
        WHERE YEAR(date) = %s AND tourney_level IS NOT NULL
        GROUP BY tourney_level ORDER BY matches DESC
    """, (year,))
    overview['levels'] = cursor.fetchall()
    connection.close()
    return overview

@st.cache_data(ttl=3600)
def get_season_leaderboard(year, min_matches=10):
    connection = get_db_connection()
    if not connection:
        return []
    cursor = connection.cursor(dictionary=True)
    cursor.execute("""
        SELECT p.player_name, p.nationality,
               SUM(CASE WHEN m.winner_name = p.player_name THEN 1 ELSE 0 END) as wins,
               SUM(CASE WHEN m.loser_name  = p.player_name THEN 1 ELSE 0 END) as losses,
               COUNT(*) as total,
               ROUND(SUM(CASE WHEN m.winner_name = p.player_name THEN 1 ELSE 0 END) / COUNT(*) * 100, 1) as win_pct,
               ps.elo_overall, ps.current_rank
        FROM players p
        JOIN matches m ON (m.winner_name = p.player_name OR m.loser_name = p.player_name)
        JOIN player_stats ps ON ps.player_id = p.player_id
        WHERE YEAR(m.date) = %s
        GROUP BY p.player_name, p.nationality, ps.elo_overall, ps.current_rank
        HAVING total >= %s
        ORDER BY win_pct DESC, wins DESC LIMIT 20
    """, (year, min_matches))
    leaders = cursor.fetchall()
    connection.close()
    return leaders


@st.cache_data(ttl=3600)
def get_surface_specialists(year):
    connection = get_db_connection()
    if not connection:
        return {}
    cursor = connection.cursor(dictionary=True)
    results = {}
    for surface in ['Hard', 'Clay', 'Grass']:
        cursor.execute("""
            SELECT p.player_name, p.nationality,
                   SUM(CASE WHEN m.winner_name = p.player_name THEN 1 ELSE 0 END) as wins,
                   COUNT(*) as total,
                   ROUND(SUM(CASE WHEN m.winner_name = p.player_name THEN 1 ELSE 0 END) / COUNT(*) * 100, 1) as win_pct
            FROM players p
            JOIN matches m ON (m.winner_name = p.player_name OR m.loser_name = p.player_name)
            WHERE YEAR(m.date) = %s AND m.surface = %s
            GROUP BY p.player_name, p.nationality
            HAVING total >= 5
            ORDER BY win_pct DESC, wins DESC LIMIT 5
        """, (year, surface))
        results[surface] = cursor.fetchall()
    connection.close()
    return results

@st.cache_data(ttl=3600)
def get_nationality_stats(year):
    connection = get_db_connection()
    if not connection:
        return []
    cursor = connection.cursor(dictionary=True)
    cursor.execute("""
        SELECT p.nationality,
               COUNT(DISTINCT p.player_name) as players,
               SUM(CASE WHEN m.winner_name = p.player_name THEN 1 ELSE 0 END) as wins,
               SUM(CASE WHEN m.loser_name  = p.player_name THEN 1 ELSE 0 END) as losses,
               COUNT(*) as total_matches
        FROM players p
        JOIN matches m ON (m.winner_name = p.player_name OR m.loser_name = p.player_name)
        WHERE YEAR(m.date) = %s AND p.nationality IS NOT NULL
        GROUP BY p.nationality
        HAVING total_matches >= 10
        ORDER BY wins DESC LIMIT 15
    """, (year,))
    stats = cursor.fetchall()
    connection.close()
    return stats

@st.cache_data(ttl=3600)
def get_monthly_activity(year):
    connection = get_db_connection()
    if not connection:
        return []
    cursor = connection.cursor(dictionary=True)
    cursor.execute("""
        SELECT MONTH(date) as month, surface, COUNT(*) as matches
        FROM matches
        WHERE YEAR(date) = %s AND surface IS NOT NULL
        GROUP BY MONTH(date), surface
        ORDER BY month, surface
    """, (year,))
    activity = cursor.fetchall()
    connection.close()
    return activity


@st.cache_data(ttl=3600)
def get_grand_slam_winners(year):
    connection = get_db_connection()
    if not connection:
        return []
    cursor = connection.cursor(dictionary=True)
    cursor.execute("""
        SELECT tournament, surface, winner_name, loser_name, score, date
        FROM matches
        WHERE YEAR(date) = %s
          AND tourney_level = 'G'
          AND round = 'F'
        ORDER BY date
    """, (year,))
    winners = cursor.fetchall()
    connection.close()
    return winners

# ── NEW: Longest and shortest matches ───────────────────────
@st.cache_data(ttl=3600)
def get_match_extremes(year):
    connection = get_db_connection()
    if not connection:
        return None, None
    cursor = connection.cursor(dictionary=True)
    cursor.execute("""
        SELECT date, tournament, surface, round,
               winner_name, loser_name, score, minutes
        FROM matches
        WHERE YEAR(date) = %s AND minutes IS NOT NULL AND minutes > 30
        ORDER BY minutes DESC LIMIT 1
    """, (year,))
    longest = cursor.fetchone()
    cursor.execute("""
        SELECT date, tournament, surface, round,
               winner_name, loser_name, score, minutes
        FROM matches
        WHERE YEAR(date) = %s AND minutes IS NOT NULL AND minutes > 30
        ORDER BY minutes ASC LIMIT 1
    """, (year,))
    shortest = cursor.fetchone()
    connection.close()
    return longest, shortest


# ── NEW: Serve leaders ───────────────────────────────────────
@st.cache_data(ttl=3600)
def get_serve_leaders(year, min_matches=20):
    connection = get_db_connection()
    if not connection:
        return {}
    cursor = connection.cursor(dictionary=True)

    # Ace leaders (total aces in the year)
    cursor.execute("""
        SELECT winner_name as player, SUM(w_ace) as stat, COUNT(*) as matches
        FROM matches
        WHERE YEAR(date) = %s AND w_ace IS NOT NULL
        GROUP BY winner_name
        HAVING matches >= %s
        UNION ALL
        SELECT loser_name, SUM(l_ace), COUNT(*)
        FROM matches
        WHERE YEAR(date) = %s AND l_ace IS NOT NULL
        GROUP BY loser_name
        HAVING COUNT(*) >= %s
    """, (year, min_matches, year, min_matches))
    rows = cursor.fetchall()

    from collections import defaultdict
    ace_totals = defaultdict(lambda: {'aces': 0, 'svpt': 0, 'matches': 0})
    # Recalculate properly with serve points
    cursor.execute("""
        SELECT winner_name as player,
               SUM(w_ace) as aces, SUM(w_svpt) as svpt, COUNT(*) as matches
        FROM matches
        WHERE YEAR(date) = %s AND w_ace IS NOT NULL AND w_svpt > 0
        GROUP BY winner_name
        UNION ALL
        SELECT loser_name,
               SUM(l_ace), SUM(l_svpt), COUNT(*)
        FROM matches
        WHERE YEAR(date) = %s AND l_ace IS NOT NULL AND l_svpt > 0
        GROUP BY loser_name
    """, (year, year))
    rows = cursor.fetchall()
    merged = defaultdict(lambda: {'aces': 0, 'svpt': 0, 'matches': 0})
    for r in rows:
        p = r['player']
        merged[p]['aces']    += r['aces'] or 0
        merged[p]['svpt']    += r['svpt'] or 0
        merged[p]['matches'] += r['matches'] or 0

    ace_rate = sorted(
        [{'player': p, 'value': round(v['aces']/v['svpt']*100, 2),
          'matches': v['matches'], 'total_aces': int(v['aces'])}
         for p, v in merged.items() if v['svpt'] > 0 and v['matches'] >= min_matches],
        key=lambda x: x['value'], reverse=True
    )[:10]

    # BP save leaders
    cursor.execute("""
        SELECT winner_name as player,
               SUM(w_bpSaved) as saved, SUM(w_bpFaced) as faced, COUNT(*) as matches
        FROM matches WHERE YEAR(date)=%s AND w_bpFaced > 0 GROUP BY winner_name
        UNION ALL
        SELECT loser_name,
               SUM(l_bpSaved), SUM(l_bpFaced), COUNT(*)
        FROM matches WHERE YEAR(date)=%s AND l_bpFaced > 0 GROUP BY loser_name
    """, (year, year))
    rows = cursor.fetchall()
    bp_merged = defaultdict(lambda: {'saved': 0, 'faced': 0, 'matches': 0})
    for r in rows:
        p = r['player']
        bp_merged[p]['saved']   += r['saved'] or 0
        bp_merged[p]['faced']   += r['faced'] or 0
        bp_merged[p]['matches'] += r['matches'] or 0

    bp_save = sorted(
        [{'player': p, 'value': round(v['saved']/v['faced']*100, 1),
          'matches': v['matches'], 'saved': int(v['saved']), 'faced': int(v['faced'])}
         for p, v in bp_merged.items() if v['faced'] >= 20 and v['matches'] >= min_matches],
        key=lambda x: x['value'], reverse=True
    )[:10]

    # 1st serve % leaders
    cursor.execute("""
        SELECT winner_name as player,
               SUM(w_1stIn) as first_in, SUM(w_svpt) as svpt, COUNT(*) as matches
        FROM matches WHERE YEAR(date)=%s AND w_svpt > 0 GROUP BY winner_name
        UNION ALL
        SELECT loser_name,
               SUM(l_1stIn), SUM(l_svpt), COUNT(*)
        FROM matches WHERE YEAR(date)=%s AND l_svpt > 0 GROUP BY loser_name
    """, (year, year))
    rows = cursor.fetchall()
    fs_merged = defaultdict(lambda: {'first_in': 0, 'svpt': 0, 'matches': 0})
    for r in rows:
        p = r['player']
        fs_merged[p]['first_in'] += r['first_in'] or 0
        fs_merged[p]['svpt']     += r['svpt'] or 0
        fs_merged[p]['matches']  += r['matches'] or 0

    first_serve = sorted(
        [{'player': p, 'value': round(v['first_in']/v['svpt']*100, 1),
          'matches': v['matches']}
         for p, v in fs_merged.items() if v['svpt'] > 0 and v['matches'] >= min_matches],
        key=lambda x: x['value'], reverse=True
    )[:10]

    connection.close()
    return {'ace_rate': ace_rate, 'bp_save': bp_save, 'first_serve': first_serve}


# ── NEW: Return / BP conversion leaders ─────────────────────
@st.cache_data(ttl=3600)
def get_return_leaders(year, min_matches=20):
    connection = get_db_connection()
    if not connection:
        return {}
    cursor = connection.cursor(dictionary=True)
    from collections import defaultdict

    # BP conversion: winner breaks loser's serve → use opponent's bp data
    # winner breaks = loser faced bp and didn't save = l_bpFaced - l_bpSaved
    cursor.execute("""
        SELECT winner_name as player,
               SUM(l_bpFaced - l_bpSaved) as converted, SUM(l_bpFaced) as chances,
               COUNT(*) as matches
        FROM matches WHERE YEAR(date)=%s AND l_bpFaced > 0 GROUP BY winner_name
        UNION ALL
        SELECT loser_name,
               SUM(w_bpFaced - w_bpSaved), SUM(w_bpFaced), COUNT(*)
        FROM matches WHERE YEAR(date)=%s AND w_bpFaced > 0 GROUP BY loser_name
    """, (year, year))
    rows = cursor.fetchall()
    bp_merged = defaultdict(lambda: {'converted': 0, 'chances': 0, 'matches': 0})
    for r in rows:
        p = r['player']
        bp_merged[p]['converted'] += r['converted'] or 0
        bp_merged[p]['chances']   += r['chances'] or 0
        bp_merged[p]['matches']   += r['matches'] or 0

    bp_conv = sorted(
        [{'player': p, 'value': round(v['converted']/v['chances']*100, 1),
          'matches': v['matches'], 'converted': int(v['converted']), 'chances': int(v['chances'])}
         for p, v in bp_merged.items() if v['chances'] >= 20 and v['matches'] >= min_matches],
        key=lambda x: x['value'], reverse=True
    )[:10]

    # 1st serve return won %: opponent's 1stIn minus opponent's 1stWon
    cursor.execute("""
        SELECT winner_name as player,
               SUM(l_1stIn - l_1stWon) as ret_won, SUM(l_1stIn) as opp_1st_in,
               COUNT(*) as matches
        FROM matches WHERE YEAR(date)=%s AND l_1stIn > 0 GROUP BY winner_name
        UNION ALL
        SELECT loser_name,
               SUM(w_1stIn - w_1stWon), SUM(w_1stIn), COUNT(*)
        FROM matches WHERE YEAR(date)=%s AND w_1stIn > 0 GROUP BY loser_name
    """, (year, year))
    rows = cursor.fetchall()
    ret_merged = defaultdict(lambda: {'ret_won': 0, 'opp_1st_in': 0, 'matches': 0})
    for r in rows:
        p = r['player']
        ret_merged[p]['ret_won']     += r['ret_won'] or 0
        ret_merged[p]['opp_1st_in']  += r['opp_1st_in'] or 0
        ret_merged[p]['matches']     += r['matches'] or 0

    ret_first = sorted(
        [{'player': p, 'value': round(v['ret_won']/v['opp_1st_in']*100, 1),
          'matches': v['matches']}
         for p, v in ret_merged.items() if v['opp_1st_in'] > 0 and v['matches'] >= min_matches],
        key=lambda x: x['value'], reverse=True
    )[:10]

    connection.close()
    return {'bp_conversion': bp_conv, 'return_first': ret_first}

# ============================================================
# FEATURE VECTOR BUILDER (84 features for LightGBM)
# ============================================================

def build_feature_vector(player_a_name, player_b_name, surface, round_name,
                          tourney_level, is_indoor, best_of):
    stats_a = get_player_stats(player_a_name)
    stats_b = get_player_stats(player_b_name)
    if not stats_a or not stats_b:
        st.error("Could not load player stats from database")
        return None

    surface_elo = {'Hard': 'elo_hard', 'Clay': 'elo_clay',
                   'Grass': 'elo_grass', 'Carpet': 'elo_carpet'}
    elo_col = surface_elo.get(surface, 'elo_hard')

    def s(v, default=0.0):
        try:
            return float(v) if v is not None else default
        except:
            return default

    hand_a = stats_a.get('hand') or 'R'
    hand_b = stats_b.get('hand') or 'R'
    both_right = 1 if hand_a == 'R' and hand_b == 'R' else 0
    both_left  = 1 if hand_a == 'L' and hand_b == 'L' else 0
    one_lefty  = 1 if (hand_a == 'L') != (hand_b == 'L') else 0

    form_a_surf  = s(stats_a.get('form_surface_20'), 0.5)
    form_b_surf  = s(stats_b.get('form_surface_20'), 0.5)
    form_a_ovr   = s(stats_a.get('wins_last_10')) / max(s(stats_a.get('matches_last_10'), 1), 1)
    form_b_ovr   = s(stats_b.get('wins_last_10')) / max(s(stats_b.get('matches_last_10'), 1), 1)

    level_key = {'G': 'form_level_G', 'M': 'form_level_M',
                 '500': 'form_level_500', '250': 'form_level_250'}
    lk = level_key.get(tourney_level, 'form_level_250')
    form_a_lvl = s(stats_a.get(lk), 0.5)
    form_b_lvl = s(stats_b.get(lk), 0.5)

    fs_a  = s(stats_a.get('first_serve_pct'), 0.60)
    fs_b  = s(stats_b.get('first_serve_pct'), 0.60)
    fsw_a = s(stats_a.get('first_serve_won_pct'), 0.70)
    fsw_b = s(stats_b.get('first_serve_won_pct'), 0.70)
    ssw_a = s(stats_a.get('second_serve_won_pct'), 0.50)
    ssw_b = s(stats_b.get('second_serve_won_pct'), 0.50)
    bp_a  = s(stats_a.get('bp_save_pct'), 0.60)
    bp_b  = s(stats_b.get('bp_save_pct'), 0.60)
    ace_a = s(stats_a.get('ace_rate'), 0.05)
    ace_b = s(stats_b.get('ace_rate'), 0.05)
    df_a  = s(stats_a.get('df_rate'), 0.03)
    df_b  = s(stats_b.get('df_rate'), 0.03)
    fsr_a = s(stats_a.get('first_serve_return_won_pct'), 0.30)
    fsr_b = s(stats_b.get('first_serve_return_won_pct'), 0.30)
    ssr_a = s(stats_a.get('second_serve_return_won_pct'), 0.50)
    ssr_b = s(stats_b.get('second_serve_return_won_pct'), 0.50)
    bpc_a = s(stats_a.get('bp_conversion_pct'), 0.40)
    bpc_b = s(stats_b.get('bp_conversion_pct'), 0.40)

    h2h = get_h2h(player_a_name, player_b_name) or {}
    if h2h and h2h.get('player_a_name') == player_a_name:
        a_h2h, b_h2h = s(h2h.get('player_a_wins')), s(h2h.get('player_b_wins'))
        a_l3,  b_l3  = s(h2h.get('player_a_wins_last_3')), s(h2h.get('player_b_wins_last_3'))
        tot_h2h, tot_l3 = s(h2h.get('total_matches')), s(h2h.get('last_3_matches'))
    elif h2h:
        a_h2h, b_h2h = s(h2h.get('player_b_wins')), s(h2h.get('player_a_wins'))
        a_l3,  b_l3  = s(h2h.get('player_b_wins_last_3')), s(h2h.get('player_a_wins_last_3'))
        tot_h2h, tot_l3 = s(h2h.get('total_matches')), s(h2h.get('last_3_matches'))
    else:
        a_h2h = b_h2h = a_l3 = b_l3 = tot_h2h = tot_l3 = 0.0

    level_ord = {'G': 4, 'M': 3, '500': 2, '250': 1, 'O': 3}.get(tourney_level, 1)

    all_rounds = ['BR', 'F', 'QF', 'R128', 'R16', 'R32', 'R64', 'SF']
    round_map  = {'Final': 'F', 'Semifinals': 'SF', 'Quarterfinals': 'QF',
                  '4th Round': 'R16', '3rd Round': 'R32',
                  '2nd Round': 'R64', '1st Round': 'R128', 'Round Robin': 'RR'}
    r = round_map.get(round_name, round_name)
    round_dummies   = {f'round_{rd}': (1 if r == rd else 0) for rd in all_rounds}
    surface_dummies = {'surface_Clay':  1 if surface == 'Clay'  else 0,
                       'surface_Grass': 1 if surface == 'Grass' else 0,
                       'surface_Hard':  1 if surface == 'Hard'  else 0}

    elo_a_ovr  = s(stats_a.get('elo_overall'), 1500)
    elo_b_ovr  = s(stats_b.get('elo_overall'), 1500)
    elo_a_surf = s(stats_a.get(elo_col), 1500)
    elo_b_surf = s(stats_b.get(elo_col), 1500)

    inactive_a = 1 if s(stats_a.get('days_since_last_match'), 999) > 30 else 0
    inactive_b = 1 if s(stats_b.get('days_since_last_match'), 999) > 30 else 0

    features = {
        'best_of':                              float(best_of),
        'player_a_rank':                        s(stats_a.get('current_rank'), 200),
        'player_a_points':                      s(stats_a.get('current_rank_points')),
        'player_a_height':                      s(stats_a.get('height'), 185),
        'player_a_is_teen':    1 if s(stats_a.get('age'), 25) < 20 else 0,
        'player_a_is_veteran': 1 if s(stats_a.get('age'), 25) > 35 else 0,
        'player_b_rank':                        s(stats_b.get('current_rank'), 200),
        'player_b_points':                      s(stats_b.get('current_rank_points')),
        'player_b_height':                      s(stats_b.get('height'), 185),
        'player_b_is_teen':    1 if s(stats_b.get('age'), 25) < 20 else 0,
        'player_b_is_veteran': 1 if s(stats_b.get('age'), 25) > 35 else 0,
        'both_righthanded':                     float(both_right),
        'both_lefthanded':                      float(both_left),
        'one_lefty':                            float(one_lefty),
        'rank_diff':                            s(stats_a.get('current_rank'), 200) - s(stats_b.get('current_rank'), 200),
        'points_diff':                          s(stats_a.get('current_rank_points')) - s(stats_b.get('current_rank_points')),
        'height_diff':                          s(stats_a.get('height'), 185) - s(stats_b.get('height'), 185),
        'player_a_form_overall':                form_a_ovr,
        'player_b_form_overall':                form_b_ovr,
        'player_a_form_surface':                form_a_surf,
        'player_b_form_surface':                form_b_surf,
        'player_a_first_serve_pct':             fs_a,
        'player_b_first_serve_pct':             fs_b,
        'player_a_first_serve_won_pct':         fsw_a,
        'player_b_first_serve_won_pct':         fsw_b,
        'player_a_second_serve_won_pct':        ssw_a,
        'player_b_second_serve_won_pct':        ssw_b,
        'player_a_bp_save_pct':                 bp_a,
        'player_b_bp_save_pct':                 bp_b,
        'player_a_ace_rate':                    ace_a,
        'player_b_ace_rate':                    ace_b,
        'player_a_df_rate':                     df_a,
        'player_b_df_rate':                     df_b,
        'first_serve_pct_diff':                 fs_a   - fs_b,
        'first_serve_won_pct_diff':             fsw_a  - fsw_b,
        'second_serve_won_pct_diff':            ssw_a  - ssw_b,
        'bp_save_pct_diff':                     bp_a   - bp_b,
        'ace_rate_diff':                        ace_a  - ace_b,
        'df_rate_diff':                         df_a   - df_b,
        'player_a_inactive':                    float(inactive_a),
        'player_b_inactive':                    float(inactive_b),
        'total_h2h':                            tot_h2h,
        'player_a_h2h_wins':                    a_h2h,
        'player_b_h2h_wins':                    b_h2h,
        'tourney_level_ordinal':                float(level_ord),
        'is_indoor':                            float(1 if is_indoor else 0),
        'player_a_last3_h2h_wins':              a_l3,
        'player_b_last3_h2h_wins':              b_l3,
        'total_last3_h2h':                      tot_l3,
        'player_a_matches_last15d':             s(stats_a.get('matches_last_15d')),
        'player_b_matches_last15d':             s(stats_b.get('matches_last_15d')),
        'last3_h2h_diff':                       a_l3   - b_l3,
        'matches_last15d_diff':                 s(stats_a.get('matches_last_15d')) - s(stats_b.get('matches_last_15d')),
        'player_a_elo_overall':                 elo_a_ovr,
        'player_b_elo_overall':                 elo_b_ovr,
        'player_a_elo_surface':                 elo_a_surf,
        'player_b_elo_surface':                 elo_b_surf,
        'elo_diff_overall':                     elo_a_ovr  - elo_b_ovr,
        'elo_diff_surface':                     elo_a_surf - elo_b_surf,
        'player_a_form_surface_20':             form_a_surf,
        'player_b_form_surface_20':             form_b_surf,
        'form_surface_20_diff':                 form_a_surf - form_b_surf,
        'player_a_first_serve_return_won_pct':  fsr_a,
        'player_b_first_serve_return_won_pct':  fsr_b,
        'player_a_second_serve_return_won_pct': ssr_a,
        'player_b_second_serve_return_won_pct': ssr_b,
        'player_a_bp_conversion_pct':           bpc_a,
        'player_b_bp_conversion_pct':           bpc_b,
        'first_serve_return_won_pct_diff':      fsr_a  - fsr_b,
        'second_serve_return_won_pct_diff':     ssr_a  - ssr_b,
        'bp_conversion_pct_diff':               bpc_a  - bpc_b,
        'player_a_form_level':                  form_a_lvl,
        'player_b_form_level':                  form_b_lvl,
        'form_level_diff':                      form_a_lvl - form_b_lvl,
        **surface_dummies,
        **round_dummies,
    }

    df = pd.DataFrame([features])
    df = df[model_data['features']]
    return df

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🎾 Tennis Predictor")
st.sidebar.markdown(f"**LightGBM** | Test: **{model_data['metrics']['test_acc']:.2%}**")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigation", [
    "🏠 Home",
    "🔮 Predict Match",
    "👤 Player Profile",
    "📅 Season Dashboard",
    "📊 Model Info",
    "ℹ️ About"
])

st.sidebar.markdown("---")
st.sidebar.markdown("**Data Management**")
if st.sidebar.button("🔄 Check for New Matches"):
    with st.spinner("Updating..."):
        try:
            from data_updater import run_update
            updated = run_update(year=datetime.now().year)
            if updated:
                st.sidebar.success("✅ Updated!")
                st.cache_data.clear()
                st.rerun()
            else:
                st.sidebar.info("Already up to date")
        except Exception as e:
            st.sidebar.error(f"Update failed: {e}")

# ============================================================
# PAGE 1: HOME
# ============================================================

if page == "🏠 Home":
    st.title("🎾 ATP Tennis Match Predictor")
    st.markdown("### Production ML System — LightGBM with Enriched Features")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎯 Test Accuracy", f"{model_data['metrics']['test_acc']:.2%}")
        st.caption("2025 unseen data")
    with col2:
        st.metric("📈 Validation AUC", f"{model_data['metrics']['val_auc']:.4f}")
        st.caption("Discrimination power")
    with col3:
        st.metric("🏆 Features", model_data['metrics']['n_features'])
        st.caption("Enriched feature set")
    with col4:
        st.metric("⚖️ Overfit Gap", f"{model_data['metrics']['overfit_gap']:.2%}")
        st.caption("Train vs Val")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🧠 What drives predictions")
        st.markdown("""
        - **Elo ratings** (variable K-factor, surface-specific)
        - **Tournament-level form** (Slam vs 250 specialization)
        - **Extended surface form** (20-match window)
        - **Serve & return stats** (ace rate, BP save %, 1st serve %)
        - **Fatigue** (matches in last 15 days)
        - **H2H history** (career + last 3 meetings)
        - **Physical** (age, height differentials)
        """)
    with col2:
        st.markdown("### 📊 Model Performance")
        m = model_data['metrics']
        df_perf = pd.DataFrame({
            'Split':    ['Train', 'Validation', 'Test'],
            'Accuracy': [f"{m['train_acc']:.2%}", f"{m['val_acc']:.2%}", f"{m['test_acc']:.2%}"],
            'AUC':      [f"{m['train_auc']:.4f}", f"{m['val_auc']:.4f}", f"{m['test_auc']:.4f}"]
        })
        st.dataframe(df_perf, hide_index=True, use_container_width=True)
        st.markdown("**Best params:**")
        for k, v in model_data['params'].items():
            st.caption(f"`{k}`: {v}")

    st.markdown("---")
    st.info("👈 Use the sidebar to predict matches, explore player profiles, or view the season dashboard")

# ============================================================
# PAGE 2: PREDICT MATCH
# ============================================================

elif page == "🔮 Predict Match":
    st.title("🔮 Match Prediction")
    st.markdown("---")

    players = get_players_list()
    if not players:
        st.error("Could not load players from database")
        st.stop()

    st.subheader("🎾 Match Setup")
    col1, col2 = st.columns(2)
    with col1:
        default_a = players.index('Sinner J.') if 'Sinner J.' in players else 0
        player_a = st.selectbox("Player A", players, index=default_a, key='pa')
    with col2:
        default_b = players.index('Alcaraz C.') if 'Alcaraz C.' in players else 1
        player_b = st.selectbox("Player B", players, index=default_b, key='pb')

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        surface = st.selectbox("Surface", ['Hard', 'Clay', 'Grass'])
    with col2:
        tourney_level = st.selectbox("Level", ['G', 'M', '500', '250'],
            format_func=lambda x: {'G': 'Grand Slam', 'M': 'Masters', '500': 'ATP 500', '250': 'ATP 250'}[x])
    with col3:
        round_name = st.selectbox("Round", ['1st Round', '2nd Round', '3rd Round',
                                             '4th Round', 'Quarterfinals', 'Semifinals', 'Final'])
    with col4:
        best_of  = st.selectbox("Best of", [3, 5])
        is_indoor = st.checkbox("Indoor")

    if player_a == player_b:
        st.warning("⚠️ Please select different players")
        st.stop()

    st.markdown("---")

    stats_a = get_player_stats(player_a)
    stats_b = get_player_stats(player_b)
    if not stats_a or not stats_b:
        st.error("Could not load player stats")
        st.stop()

    def safe_int(v, d=0):
        return int(v) if v is not None else d

    # Player comparison
    st.subheader("📊 Player Comparison")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"### 👤 {player_a}")
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Rank",    f"#{safe_int(stats_a['current_rank'])}")
            st.metric("Elo",     f"{safe_int(stats_a['elo_overall'], 1500)}")
        with m2:
            st.metric("Age",     f"{safe_int(stats_a['age'])}")
            form_a = safe_int(stats_a['wins_last_10']) / max(safe_int(stats_a['matches_last_10']), 1) * 100
            st.metric("Form L10", f"{form_a:.0f}%")
    with col2:
        st.markdown(f"### 👤 {player_b}")
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Rank",    f"#{safe_int(stats_b['current_rank'])}")
            st.metric("Elo",     f"{safe_int(stats_b['elo_overall'], 1500)}")
        with m2:
            st.metric("Age",     f"{safe_int(stats_b['age'])}")
            form_b = safe_int(stats_b['wins_last_10']) / max(safe_int(stats_b['matches_last_10']), 1) * 100
            st.metric("Form L10", f"{form_b:.0f}%")

    st.markdown("---")

    # Season record
    current_year = datetime.now().year
    st.subheader(f"📅 {current_year} Season Record")
    col1, col2 = st.columns(2)
    with col1:
        wa, la = get_season_record(player_a, current_year)
        ta = wa + la
        st.metric(player_a, f"{wa}-{la}", f"{wa/ta*100:.0f}% win rate" if ta > 0 else "No matches yet")
    with col2:
        wb, lb = get_season_record(player_b, current_year)
        tb = wb + lb
        st.metric(player_b, f"{wb}-{lb}", f"{wb/tb*100:.0f}% win rate" if tb > 0 else "No matches yet")

    st.markdown("---")

    # Readiness
    st.subheader("⏱️ Player Readiness")
    col1, col2 = st.columns(2)
    with col1:
        days_a = get_days_since_last_match(player_a)
        (st.warning if days_a > 30 else st.success)(
            f"{'⚠️' if days_a > 30 else '✅'} {player_a}: **{days_a} days** since last match")
    with col2:
        days_b = get_days_since_last_match(player_b)
        (st.warning if days_b > 30 else st.success)(
            f"{'⚠️' if days_b > 30 else '✅'} {player_b}: **{days_b} days** since last match")

    st.markdown("---")

    # Recent form
    st.subheader("🏆 Recent Form — Last 5 Matches")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**{player_a}**")
        for m in get_recent_matches(player_a, 5):
            opp  = m['loser_name'] if m['result'] == 'W' else m['winner_name']
            icon = '✅' if m['result'] == 'W' else '❌'
            st.markdown(f"{icon} **{m['result']}** vs {opp}")
            st.caption(f"{m['date'].strftime('%Y-%m-%d')} • {m['tournament']} • {m['surface']} • {m['score'] or 'N/A'}")
    with col2:
        st.markdown(f"**{player_b}**")
        for m in get_recent_matches(player_b, 5):
            opp  = m['loser_name'] if m['result'] == 'W' else m['winner_name']
            icon = '✅' if m['result'] == 'W' else '❌'
            st.markdown(f"{icon} **{m['result']}** vs {opp}")
            st.caption(f"{m['date'].strftime('%Y-%m-%d')} • {m['tournament']} • {m['surface']} • {m['score'] or 'N/A'}")

    st.markdown("---")

    # Surface record this year
    st.subheader(f"🏟️ {current_year} Record on {surface}")
    col1, col2 = st.columns(2)
    with col1:
        ws, ls = get_surface_record(player_a, surface, current_year)
        ts = ws + ls
        st.metric(player_a, f"{ws}-{ls}", f"{ws/ts*100:.0f}%" if ts > 0 else "No matches")
    with col2:
        ws, ls = get_surface_record(player_b, surface, current_year)
        ts = ws + ls
        st.metric(player_b, f"{ws}-{ls}", f"{ws/ts*100:.0f}%" if ts > 0 else "No matches")

    st.markdown("---")

    # H2H
    h2h      = get_h2h(player_a, player_b)
    a_wins_h = b_wins_h = 0
    if h2h and h2h['total_matches'] > 0:
        st.subheader("🤝 Head-to-Head")
        if h2h['player_a_name'] == player_a:
            a_wins_h, b_wins_h = h2h['player_a_wins'], h2h['player_b_wins']
        else:
            a_wins_h, b_wins_h = h2h['player_b_wins'], h2h['player_a_wins']

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Career H2H", f"{a_wins_h} - {b_wins_h}")
        with col2:
            leader = player_a if a_wins_h > b_wins_h else (player_b if b_wins_h > a_wins_h else "Even")
            st.metric("Leader", leader)
        with col3:
            st.metric("Total Meetings", h2h['total_matches'])

        with st.expander(f"📋 Full H2H History ({h2h['total_matches']} matches)"):
            h2h_matches = get_h2h_matches(player_a, player_b)
            if h2h_matches:
                rows = [{'Date': m['date'].strftime('%Y-%m-%d'),
                         'Winner': f"✅ {m['winner_name']}",
                         'Tournament': m['tournament'] or 'N/A',
                         'Surface': m['surface'] or 'N/A',
                         'Round': m['round'] or 'N/A',
                         'Score': m['score'] or 'N/A'} for m in h2h_matches]
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("**By Surface:**")
                    for surf in ['Hard', 'Clay', 'Grass']:
                        sm = [m for m in h2h_matches if m['surface'] == surf]
                        if sm:
                            aw = sum(1 for m in sm if m['winner_name'] == player_a)
                            st.caption(f"{surf}: {player_a} {aw}-{len(sm)-aw} {player_b}")
                with col2:
                    st.markdown("**Grand Slams:**")
                    slams  = ['Australian Open', 'Roland Garros', 'Wimbledon', 'US Open']
                    slam_m = [m for m in h2h_matches if m['tournament'] in slams]
                    if slam_m:
                        aw = sum(1 for m in slam_m if m['winner_name'] == player_a)
                        st.caption(f"{player_a} {aw}-{len(slam_m)-aw} {player_b}")
                    else:
                        st.caption("No meetings")
                with col3:
                    st.markdown("**Last 3:**")
                    r3 = h2h_matches[:3]
                    aw = sum(1 for m in r3 if m['winner_name'] == player_a)
                    st.caption(f"{player_a} {aw}-{len(r3)-aw} {player_b}")
    else:
        st.info("ℹ️ No head-to-head history between these players")

    st.markdown("---")

    # Prediction
    if st.button("🔮 Predict Match Outcome", type="primary", use_container_width=True):
        with st.spinner("Running LightGBM prediction..."):
            features = build_feature_vector(
                player_a, player_b, surface, round_name,
                tourney_level, is_indoor, best_of
            )
            if features is None:
                st.stop()

            proba  = model_data['model'].predict_proba(features)[0][1]
            prob_a = proba
            prob_b = 1 - proba
            winner   = player_a if prob_a > prob_b else player_b
            win_prob = max(prob_a, prob_b)
            confidence = "🔴 High" if win_prob > 0.70 else ("🟡 Medium" if win_prob > 0.60 else "🟢 Low")

        st.success("✅ Prediction Complete!")
        st.markdown("---")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🏆 Predicted Winner", winner)
        with col2:
            st.metric("📊 Win Probability", f"{win_prob:.1%}")
        with col3:
            st.metric("🎯 Confidence", confidence)

        fig = go.Figure(go.Bar(
            x=[prob_a * 100, prob_b * 100],
            y=[player_a, player_b],
            orientation='h',
            text=[f"{prob_a:.1%}", f"{prob_b:.1%}"],
            textposition='auto',
            marker_color=['#2196F3' if prob_a >= prob_b else '#90CAF9',
                          '#2196F3' if prob_b > prob_a  else '#90CAF9']
        ))
        fig.update_layout(xaxis_title="Win Probability (%)", height=180,
                          xaxis_range=[0, 100], showlegend=False,
                          margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("🔍 Key Factors"):
            elo_diff  = float(stats_a['elo_overall'] or 1500) - float(stats_b['elo_overall'] or 1500)
            rank_diff = float(stats_a['current_rank'] or 200) - float(stats_b['current_rank'] or 200)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Elo Advantage", f"{abs(elo_diff):.0f} pts",
                          delta=f"{player_a if elo_diff > 0 else player_b} leads")
            with col2:
                st.metric("Rank Advantage", f"{abs(rank_diff):.0f} spots",
                          delta=f"{player_a if rank_diff < 0 else player_b} leads")
            with col3:
                st.metric("Career H2H", f"{a_wins_h}-{b_wins_h}")

# ============================================================
# PAGE 3: PLAYER PROFILE
# ============================================================

elif page == "👤 Player Profile":
    st.title("👤 Player Profile")
    st.markdown("---")

    players = get_players_list()
    default_idx = players.index('Sinner J.') if 'Sinner J.' in players else 0
    selected = st.selectbox("Select Player", players, index=default_idx)

    stats = get_player_stats(selected)
    if not stats:
        st.error(f"Could not load stats for {selected}")
        st.stop()

    def safe_int(v, d=0):
        return int(v) if v is not None else d

    st.markdown("---")
    st.markdown(f"## {selected}")
    nat  = stats.get('nationality') or '?'
    hand = 'Right-handed' if stats.get('hand') == 'R' else ('Left-handed' if stats.get('hand') == 'L' else '?')
    ht   = stats.get('height') or '?'
    st.caption(f"🌍 {nat} | {hand} | {ht} cm")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("ATP Rank", f"#{stats['current_rank'] or '?'}")
    with col2:
        st.metric("Elo Rating", f"{safe_int(stats['elo_overall'], 1500)}")
    with col3:
        st.metric("Age", f"{safe_int(stats['age'])}")
    with col4:
        form = safe_int(stats['wins_last_10']) / max(safe_int(stats['matches_last_10']), 1) * 100
        st.metric("Form (L10)", f"{form:.0f}%",
                  delta=f"{stats['wins_last_10'] or 0}W-{(stats['matches_last_10'] or 0)-(stats['wins_last_10'] or 0)}L")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎾 Elo by Surface")
        elo_vals = {'Overall': safe_int(stats['elo_overall'], 1500),
                    'Hard':    safe_int(stats['elo_hard'],    1500),
                    'Clay':    safe_int(stats['elo_clay'],    1500),
                    'Grass':   safe_int(stats['elo_grass'],   1500)}
        fig = go.Figure(go.Bar(
            y=list(elo_vals.keys()), x=list(elo_vals.values()),
            orientation='h',
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
            text=list(elo_vals.values()), textposition='auto'
        ))
        fig.update_layout(height=220, xaxis_title="Elo", showlegend=False,
                          margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🎯 Serve & Return Stats")
        if stats.get('first_serve_pct'):
            serve_data = {
                '1st Serve %':         f"{stats['first_serve_pct']*100:.1f}%",
                '1st Serve Won %':     f"{stats['first_serve_won_pct']*100:.1f}%",
                '2nd Serve Won %':     f"{stats['second_serve_won_pct']*100:.1f}%",
                'BP Save %':           f"{stats['bp_save_pct']*100:.1f}%",
                'Ace Rate':            f"{stats['ace_rate']*100:.1f}%",
                'DF Rate':             f"{stats['df_rate']*100:.1f}%",
                '1st Return Won %':    f"{stats['first_serve_return_won_pct']*100:.1f}%",
                '2nd Return Won %':    f"{stats['second_serve_return_won_pct']*100:.1f}%",
                'BP Conversion %':     f"{stats['bp_conversion_pct']*100:.1f}%",
            }
            for label, val in serve_data.items():
                st.text(f"{label}: {val}")
        else:
            st.info("No serve stats available for this player")

    st.markdown("---")

    # Season records
    st.subheader("📅 Season-by-Season Record")
    season_rows = []
    for yr in range(2020, datetime.now().year + 1):
        w, l = get_season_record(selected, yr)
        if w + l > 0:
            season_rows.append({'Year': yr, 'W': w, 'L': l,
                                 'Total': w+l, 'Win%': f"{w/(w+l)*100:.1f}%"})
    if season_rows:
        st.dataframe(pd.DataFrame(season_rows), hide_index=True, use_container_width=True)
    else:
        st.info("No season data available")

    st.markdown("---")

    # Last 10 matches
    st.subheader("🏆 Last 10 Matches")
    recent = get_recent_matches(selected, 10)
    if recent:
        rows = [{'Date':       m['date'].strftime('%Y-%m-%d'),
                 'Result':     '✅ W' if m['result'] == 'W' else '❌ L',
                 'Opponent':   m['loser_name'] if m['result'] == 'W' else m['winner_name'],
                 'Tournament': m['tournament'] or 'N/A',
                 'Surface':    m['surface'] or 'N/A',
                 'Round':      m['round'] or 'N/A',
                 'Score':      m['score'] or 'N/A'} for m in recent]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.info("No recent matches found")

# ============================================================
# PAGE 4: SEASON DASHBOARD
# ============================================================
elif page == "📅 Season Dashboard":
    st.title("📅 Season Dashboard")
    st.markdown("---")

    available_years = list(range(2000, datetime.now().year + 1))
    year = st.selectbox("Select Season", available_years,
                        index=available_years.index(datetime.now().year))
    st.markdown("---")

    # Overview metrics
    overview = get_season_overview(year)
    if not overview:
        st.error("Could not load season data")
        st.stop()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎾 Total Matches",  f"{overview.get('total_matches', 0):,}")
    with col2:
        st.metric("🏟️ Tournaments",    f"{overview.get('total_tournaments', 0):,}")
    with col3:
        st.metric("👤 Players",        f"{overview.get('total_players', 0):,}")
    with col4:
        avg_dur = overview.get('avg_duration')
        st.metric("⏱️ Avg Duration",   f"{int(avg_dur)} min" if avg_dur else "N/A")

    st.markdown("---")

    # Grand Slam winners
    st.subheader(f"🏆 Grand Slam Winners {year}")
    slam_winners = get_grand_slam_winners(year)
    slam_order = ['Australian Open', 'Roland Garros', 'Wimbledon', 'US Open']
    slam_surface = {'Australian Open': '🔵 Hard', 'Roland Garros': '🟠 Clay',
                    'Wimbledon': '🟢 Grass', 'US Open': '🔵 Hard'}

    if slam_winners:
        slam_dict = {s['tournament']: s for s in slam_winners}
        cols = st.columns(4)
        for col, slam in zip(cols, slam_order):
            with col:
                data = slam_dict.get(slam)
                st.markdown(f"**{slam}**")
                st.caption(slam_surface.get(slam, ''))
                if data:
                    st.markdown(f"🏆 **{data['winner_name']}**")
                    st.caption(f"def. {data['loser_name']}")
                    st.caption(f"{data['score']}")
                else:
                    st.caption("Not yet played")
    else:
        st.info(f"No Grand Slam data available for {year}")

    st.markdown("---")

    # Monthly activity by surface
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📅 Monthly Activity by Surface")
        monthly = get_monthly_activity(year)
        if monthly:
            month_names = ['Jan','Feb','Mar','Apr','May','Jun',
                           'Jul','Aug','Sep','Oct','Nov','Dec']
            surface_colors = {'Hard': '#2196F3', 'Clay': '#FF5722',
                              'Grass': '#4CAF50', 'Carpet': '#9C27B0'}

            # Pivot the data
            from collections import defaultdict
            pivot = defaultdict(lambda: defaultdict(int))
            surfaces_found = set()
            for row in monthly:
                pivot[row['month']][row['surface']] += row['matches']
                surfaces_found.add(row['surface'])

            months_present = sorted(pivot.keys())
            fig = go.Figure()
            for surf in sorted(surfaces_found):
                fig.add_trace(go.Bar(
                    name=surf,
                    x=[month_names[m-1] for m in months_present],
                    y=[pivot[m].get(surf, 0) for m in months_present],
                    marker_color=surface_colors.get(surf, '#999'),
                    text=[pivot[m].get(surf, 0) if pivot[m].get(surf, 0) > 0 else ''
                          for m in months_present],
                    textposition='inside'
                ))
            fig.update_layout(barmode='stack', height=320,
                              xaxis_title="Month", yaxis_title="Matches",
                              legend_title="Surface",
                              margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🏟️ Matches by Surface")
        surfaces = overview.get('surfaces', [])
        if surfaces:
            colors = {'Hard': '#2196F3', 'Clay': '#FF5722',
                      'Grass': '#4CAF50', 'Carpet': '#9C27B0'}
            fig = go.Figure(go.Pie(
                labels=[s['surface'] for s in surfaces],
                values=[s['matches'] for s in surfaces],
                marker_colors=[colors.get(s['surface'], '#999') for s in surfaces],
                hole=0.4,
                textinfo='label+percent+value'
            ))
            fig.update_layout(height=320, margin=dict(l=0, r=0, t=0, b=0),
                              showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Match extremes
    st.subheader(f"⏱️ Match Extremes {year}")
    longest, shortest = get_match_extremes(year)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🕰️ Longest Match**")
        if longest:
            st.metric("Duration", f"{longest['minutes']} min")
            st.markdown(f"**{longest['winner_name']}** def. {longest['loser_name']}")
            st.caption(f"{longest['score']}")
            st.caption(f"{longest['date'].strftime('%Y-%m-%d')} • {longest['tournament']} • {longest['surface']} • {longest['round']}")
        else:
            st.info("No data")
    with col2:
        st.markdown("**⚡ Shortest Match**")
        if shortest:
            st.metric("Duration", f"{shortest['minutes']} min")
            st.markdown(f"**{shortest['winner_name']}** def. {shortest['loser_name']}")
            st.caption(f"{shortest['score']}")
            st.caption(f"{shortest['date'].strftime('%Y-%m-%d')} • {shortest['tournament']} • {shortest['surface']} • {shortest['round']}")
        else:
            st.info("No data")

    st.markdown("---")

    # Win leaderboard
    st.subheader(f"🏆 {year} Win Leaderboard")
    st.caption("Minimum 10 matches played. Sorted by wins. Win% = wins / (wins + losses).")
    leaders = get_season_leaderboard(year)
    if leaders:
        df_lead = pd.DataFrame([{
            '#':       i+1,
            'Player':  r['player_name'],
            'Country': r['nationality'] or '?',
            'W':       r['wins'],
            'L':       r['losses'],
            'Total':   r['total'],
            'Win%':    f"{r['win_pct']}%",
            'Elo':     f"{int(r['elo_overall'] or 1500)}",
            'Rank':    r['current_rank'] or '?'
        } for i, r in enumerate(leaders)])
        st.dataframe(df_lead, hide_index=True, use_container_width=True)

    st.markdown("---")

    # Serve & return leaders
    st.subheader(f"🎾 Statistical Leaders {year}")
    st.caption("Minimum 20 matches. Calculated from raw match stats.")

    serve_data   = get_serve_leaders(year)
    return_data  = get_return_leaders(year)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**💥 Ace Rate Leaders** (aces per 100 serve pts)")
        ace_leaders = serve_data.get('ace_rate', [])
        if ace_leaders:
            for i, p in enumerate(ace_leaders[:8]):
                st.markdown(f"{i+1}. **{p['player']}** — {p['value']}%")
                st.caption(f"{p['total_aces']:,} total aces • {p['matches']} matches")
        else:
            st.info("No data")

    with col2:
        st.markdown("**🛡️ BP Save % Leaders** (saved / faced)")
        bp_save = serve_data.get('bp_save', [])
        if bp_save:
            for i, p in enumerate(bp_save[:8]):
                st.markdown(f"{i+1}. **{p['player']}** — {p['value']}%")
                st.caption(f"{p['saved']}/{p['faced']} BPs • {p['matches']} matches")
        else:
            st.info("No data")

    with col3:
        st.markdown("**📊 1st Serve % Leaders**")
        fs_leaders = serve_data.get('first_serve', [])
        if fs_leaders:
            for i, p in enumerate(fs_leaders[:8]):
                st.markdown(f"{i+1}. **{p['player']}** — {p['value']}%")
                st.caption(f"{p['matches']} matches")
        else:
            st.info("No data")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🔁 BP Conversion % Leaders** (breaks / chances)")
        bp_conv = return_data.get('bp_conversion', [])
        if bp_conv:
            for i, p in enumerate(bp_conv[:8]):
                st.markdown(f"{i+1}. **{p['player']}** — {p['value']}%")
                st.caption(f"{p['converted']}/{p['chances']} BPs • {p['matches']} matches")
        else:
            st.info("No data")

    with col2:
        st.markdown("**↩️ 1st Serve Return Won % Leaders**")
        ret_first = return_data.get('return_first', [])
        if ret_first:
            for i, p in enumerate(ret_first[:8]):
                st.markdown(f"{i+1}. **{p['player']}** — {p['value']}%")
                st.caption(f"{p['matches']} matches")
        else:
            st.info("No data")

    st.markdown("---")

    # Nationality breakdown
    st.subheader(f"🌍 Nations Performance {year}")
    nat_stats = get_nationality_stats(year)
    if nat_stats:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Wins',
            x=[n['nationality'] for n in nat_stats],
            y=[n['wins'] for n in nat_stats],
            marker_color='#4CAF50'
        ))
        fig.add_trace(go.Bar(
            name='Losses',
            x=[n['nationality'] for n in nat_stats],
            y=[n['losses'] for n in nat_stats],
            marker_color='#F44336'
        ))
        fig.update_layout(barmode='stack', height=350,
                          xaxis_title="Country", yaxis_title="Matches",
                          margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Surface specialists
    st.subheader(f"🎾 Surface Specialists {year}")
    st.caption("Min 5 matches on surface. Ranked by win %.")
    specialists = get_surface_specialists(year)
    col1, col2, col3 = st.columns(3)
    for col, surf in zip([col1, col2, col3], ['Hard', 'Clay', 'Grass']):
        with col:
            st.markdown(f"**{surf}**")
            data = specialists.get(surf, [])
            if data:
                for i, p in enumerate(data):
                    st.markdown(f"{i+1}. **{p['player_name']}** ({p['nationality'] or '?'})")
                    st.caption(f"{p['wins']}W / {p['total']} matches — {p['win_pct']}%")
            else:
                st.info("No data")
# ============================================================
# PAGE 5: MODEL INFO
# ============================================================

elif page == "📊 Model Info":
    st.title("📊 Model Information")
    st.markdown("---")

    m = model_data['metrics']
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Test Accuracy",  f"{m['test_acc']:.2%}")
        st.metric("Val Accuracy",   f"{m['val_acc']:.2%}")
        st.metric("Train Accuracy", f"{m['train_acc']:.2%}")
    with col2:
        st.metric("Test AUC",  f"{m['test_auc']:.4f}")
        st.metric("Val AUC",   f"{m['val_auc']:.4f}")
        st.metric("Train AUC", f"{m['train_auc']:.4f}")
    with col3:
        st.metric("Overfit Gap",  f"{m['overfit_gap']:.2%}")
        st.metric("Val-Test Gap", f"{m['val_test_gap']:.2%}")
        st.metric("Features",     m['n_features'])

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("⚙️ Best Hyperparameters")
        for k, v in model_data['params'].items():
            st.text(f"{k}: {v}")
    with col2:
        st.subheader(f"📋 All {m['n_features']} Features")
        for i, f in enumerate(model_data['features'], 1):
            st.text(f"{i:2d}. {f}")

# ============================================================
# PAGE 6: ABOUT
# ============================================================

elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    m = model_data['metrics']
    st.markdown(f"""
    ### ATP Tennis Match Predictor

    End-to-end ML system for ATP tennis match prediction built as a data science portfolio project.

    **Production Model: LightGBM GridSearch (Age Flags)**
    - Test Accuracy: **{m['test_acc']:.2%}** (2025 unseen data)
    - Validation AUC: **{m['val_auc']:.4f}**
    - {m['n_features']} engineered features
    - Trained on 66,703 matches (2000–2024)

    **Key Innovations:**
    - Variable K-factor Elo (tournament-weighted: GS=40, Masters=36, 500=32, 250=28)
    - Tournament-level form (captures tier specialization)
    - Extended surface form (20-match window vs standard 5)
    - Serve/return stats integration
    - Age represented as binary flags (is_teen / is_veteran) — eliminates systematic
      bias toward younger players while preserving genuine edge-case signal

    **Database Architecture (MySQL):**
    - `players` — 1,647 ATP players with nationality, hand, height, birth date
    - `matches` — 66,703 matches with full serve stats (ace, DF, BP, etc.)
    - `match_features` — 85 pre-computed features per match
    - `player_stats` — current player metrics + serve/return averages
    - `player_elo` — Elo ratings by surface
    - `h2h_history` — head-to-head records with surface breakdown

    **Tech Stack:** Python · LightGBM · MySQL · Streamlit · Plotly · pandas · scikit-learn

    **Created by:** Facundo Rabinovich
    **Date:** {m.get('date_saved', 'March 2026')}
    """)