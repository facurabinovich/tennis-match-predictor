"""
data_updater.py  -  Incremental updater for the ATP Tennis MySQL database.

Source : https://stats.tennismylife.org/api/data-files
Called : Streamlit app on session start, or manually via:
    python -c "from data_updater import run_update; run_update(2026)"

TML confirmed columns (from API 2026-03-03):
    tourney_date (int YYYYMMDD), tourney_name, surface, tourney_level,
    indoor (I/O), best_of, round, score, minutes,
    winner_id (ATP str), winner_name, winner_hand, winner_ht, winner_ioc,
    winner_age, winner_rank, winner_rank_points,
    loser_id  (ATP str), loser_name,  loser_hand,  loser_ht,  loser_ioc,
    loser_age,  loser_rank,  loser_rank_points,
    w_ace..w_bpFaced, l_ace..l_bpFaced

    NOTE: winner_age / loser_age are float values (e.g. 25.81).
    We convert these to an estimated birth_date and store in players.birth_date.
    Age is never stored directly — always calculated on the fly via TIMESTAMPDIFF.

Schema reference (DO NOT ALTER TABLES — code adapts to existing schema):
    players       : player_id, player_name, hand, height, created_at, updated_at,
                    atp_id, nationality, birth_date
    matches       : match_id, date, tournament, surface, round, best_of,
                    winner_id, winner_atp_id, winner_name, winner_rank, winner_rank_points,
                    loser_id, loser_atp_id, loser_name, loser_rank, loser_rank_points,
                    score, minutes, created_at, tournament_id, tourney_level, indoor(varchar5),
                    w_ace..w_bpFaced, l_ace..l_bpFaced
    player_elo    : elo_id, player_id, date, elo_overall, elo_hard, elo_clay, elo_grass, elo_carpet
    player_stats  : player_id, current_rank, current_rank_points, height, hand,
                    elo_overall, elo_hard, elo_clay, elo_grass, elo_carpet,
                    matches_last_10, wins_last_10, matches_last_15d,
                    last_match_date, days_since_last_match, updated_at,
                    form_surface_20, form_level_G, form_level_M, form_level_500, form_level_250,
                    first_serve_pct, first_serve_won_pct, second_serve_won_pct,
                    bp_save_pct, ace_rate, df_rate,
                    first_serve_return_won_pct, second_serve_return_won_pct, bp_conversion_pct
    h2h_history   : h2h_id, player_a_id, player_b_id, total_matches,
                    player_a_wins, player_b_wins, last_3_matches,
                    player_a_wins_last_3, player_b_wins_last_3,
                    hard_matches, player_a_wins_hard, clay_matches, player_a_wins_clay,
                    grass_matches, player_a_wins_grass, updated_at
    match_features: feature_id, match_id, player_a_id, player_b_id,
                    player_a_elo_overall, player_b_elo_overall, elo_diff_overall,
                    player_a_elo_surface, player_b_elo_surface, elo_diff_surface,
                    player_a_rank, player_b_rank, rank_diff,
                    player_a_points, player_b_points, points_diff,
                    player_a_height, player_b_height, height_diff,
                    player_a_form_overall, player_b_form_overall,
                    player_a_form_surface, player_b_form_surface,
                    player_a_inactive, player_b_inactive,
                    total_h2h, player_a_h2h_wins, player_b_h2h_wins,
                    total_last3_h2h, player_a_last3_h2h_wins, player_b_last3_h2h_wins,
                    player_a_matches_last15d, player_b_matches_last15d,
                    last3_h2h_diff, matches_last15d_diff,
                    player_a_won, created_at,
                    form_surface_20_diff,
                    player_a_form_level, player_b_form_level, form_level_diff,
                    first_serve_return_won_pct_diff, second_serve_return_won_pct_diff,
                    bp_conversion_pct_diff,
                    player_a_form_surface_20, player_b_form_surface_20
"""

import requests
import pandas as pd
import numpy as np
from io import StringIO
from datetime import datetime, date, timedelta
# from config_local import get_db_connection
from config import get_db_connection

# ── CONFIG ───────────────────────────────────────────────────
TML_API_INDEX = "https://stats.tennismylife.org/api/data-files"
TML_BASE_URL  = "https://stats.tennismylife.org"
ONGOING_FILE  = "ongoing_tourneys.csv"

K_FACTORS   = {'G': 40, 'M': 36, '500': 32, '250': 28, 'O': 30}
DEFAULT_K   = 28
ELO_DEFAULT = 1500.0

# ── HELPERS ──────────────────────────────────────────────────

def _safe(v, default=None):
    if v is None:
        return default
    try:
        if pd.isna(v):
            return default
    except (TypeError, ValueError):
        pass
    return v

def _safe_float(v, default=None):
    try:
        f = float(v)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return default

def _safe_int(v, default=None):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default

def _age_to_birthdate(age_float, match_date):
    """Estimate birth_date from TML age float (e.g. 25.81) and match date.
    Approximation: accurate to within a few days. Good enough for storage."""
    if age_float is None:
        return None
    try:
        age_days = int(float(age_float) * 365.25)
        return match_date - timedelta(days=age_days)
    except (TypeError, ValueError):
        return None

def _elo_update(elo_w, elo_l, K):
    exp_w = 1.0 / (1.0 + 10 ** ((elo_l - elo_w) / 400.0))
    return (elo_w + K * (1 - exp_w),
            elo_l + K * (0 - (1 - exp_w)))

def _k_factor(lvl):
    return K_FACTORS.get(str(lvl).strip(), DEFAULT_K)

# ── 1. FETCH INDEX ───────────────────────────────────────────

def _fetch_index():
    """Returns {filename: url} from TML API ({"count":N,"files":[...]})."""
    try:
        resp = requests.get(TML_API_INDEX, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[updater] Cannot reach TML API: {e}")
        return {}

    file_list = data.get('files') if isinstance(data, dict) else data
    if not isinstance(file_list, list):
        print(f"[updater] Unexpected API response: {type(data)}")
        return {}

    print(f"[updater] API index: {len(file_list)} files available")
    index = {}
    for item in file_list:
        name = item.get('name', '')
        url  = item.get('url', '')
        if name and url:
            index[name] = url
    return index

# ── 2. DOWNLOAD CSV ──────────────────────────────────────────

def _download_csv(url, label=''):
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        print(f"[updater] Downloaded {label}: {len(df)} rows")
        return df
    except Exception as e:
        print(f"[updater] Failed to download {label}: {e}")
        return None

# ── 3. NORMALISE ─────────────────────────────────────────────

def _normalise(df):
    """
    Map TML column names to internal names.
    indoor is kept as 'I'/'O' string to match the matches.indoor varchar(5) column.
    """
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    aliases = {
        'tourney_date':       'date',
        'tourney_name':       'tournament',
        'winner_rank_points': 'winner_pts',
        'loser_rank_points':  'loser_pts',
        'winner_id':          'winner_atp_id',
        'loser_id':           'loser_atp_id',
    }
    df = df.rename(columns={k: v for k, v in aliases.items() if k in df.columns})

    # date: TML integer 20260102
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(
            df['date'].astype(str), format='%Y%m%d', errors='coerce'
        ).dt.date

    # indoor: keep as 'I' or 'O' string — matches DB varchar(5) column
    if 'indoor' in df.columns:
        df['indoor'] = df['indoor'].apply(
            lambda v: 'I' if str(v).strip().upper() in ('I', '1', 'TRUE', 'YES') else 'O'
        )

    # surface capitalise
    if 'surface' in df.columns:
        df['surface'] = df['surface'].str.capitalize()

    df = df.dropna(subset=['winner_name', 'loser_name'])
    df['winner_name'] = df['winner_name'].astype(str).str.strip()
    df['loser_name']  = df['loser_name'].astype(str).str.strip()
    return df

# ── DB HELPERS ───────────────────────────────────────────────

def _latest_date_in_db(cursor):
    cursor.execute("SELECT MAX(date) FROM matches")
    row = cursor.fetchone()
    return row[0] if row and row[0] else date(2000, 1, 1)

def _existing_keys(cursor, since):
    cursor.execute(
        "SELECT date, winner_name, loser_name FROM matches WHERE date >= %s", (since,)
    )
    return {(r[0], r[1], r[2]) for r in cursor.fetchall()}

# ── PLAYER UPSERT ────────────────────────────────────────────

def _get_or_create_player(cursor, name, hand=None, ht=None,
                           nationality=None, atp_id=None, birth_date=None):
    cursor.execute("SELECT player_id FROM players WHERE player_name = %s", (name,))
    row = cursor.fetchone()
    if row:
        pid = row[0]
        upd, vals = [], []
        if hand:        upd.append("hand        = COALESCE(hand, %s)");        vals.append(hand)
        if ht:          upd.append("height      = COALESCE(height, %s)");      vals.append(_safe_int(ht))
        if nationality: upd.append("nationality = COALESCE(nationality, %s)"); vals.append(nationality)
        if atp_id:      upd.append("atp_id      = COALESCE(atp_id, %s)");      vals.append(str(atp_id))
        if birth_date:  upd.append("birth_date  = COALESCE(birth_date, %s)");  vals.append(birth_date)
        if upd:
            vals.append(pid)
            cursor.execute(f"UPDATE players SET {', '.join(upd)} WHERE player_id = %s", vals)
        return pid

    cursor.execute(
        "INSERT INTO players (player_name, hand, height, nationality, atp_id, birth_date) "
        "VALUES (%s,%s,%s,%s,%s,%s)",
        (name, _safe(hand), _safe_int(ht), _safe(nationality),
         str(atp_id) if atp_id else None, birth_date)
    )
    return cursor.lastrowid

# ── ELO ──────────────────────────────────────────────────────

def _get_elos(cursor, player_id):
    cursor.execute(
        "SELECT elo_overall, elo_hard, elo_clay, elo_grass, elo_carpet "
        "FROM player_elo WHERE player_id = %s", (player_id,)
    )
    row = cursor.fetchone()
    if row:
        return {
            'overall': float(row[0] or ELO_DEFAULT), 'Hard':   float(row[1] or ELO_DEFAULT),
            'Clay':    float(row[2] or ELO_DEFAULT),  'Grass':  float(row[3] or ELO_DEFAULT),
            'Carpet':  float(row[4] or ELO_DEFAULT),
        }
    return {s: ELO_DEFAULT for s in ('overall', 'Hard', 'Clay', 'Grass', 'Carpet')}

def _save_elo(cursor, player_id, elos, match_date):
    cursor.execute("""
        INSERT INTO player_elo
            (player_id, date, elo_overall, elo_hard, elo_clay, elo_grass, elo_carpet)
        VALUES (%s,%s,%s,%s,%s,%s,%s)
        ON DUPLICATE KEY UPDATE
            date=VALUES(date), elo_overall=VALUES(elo_overall),
            elo_hard=VALUES(elo_hard), elo_clay=VALUES(elo_clay),
            elo_grass=VALUES(elo_grass), elo_carpet=VALUES(elo_carpet)
    """, (player_id, match_date,
          elos['overall'], elos['Hard'], elos['Clay'], elos['Grass'], elos['Carpet']))

# ── PLAYER STATS ─────────────────────────────────────────────

def _recalc_player_stats(cursor, player_id, player_name, match_date, elos):
    cursor.execute("""
        SELECT date, winner_name, loser_name, surface, tourney_level,
               winner_rank, loser_rank, winner_rank_points, loser_rank_points,
               w_ace, w_df, w_svpt, w_1stIn, w_1stWon, w_2ndWon,
               w_SvGms, w_bpSaved, w_bpFaced,
               l_ace, l_df, l_svpt, l_1stIn, l_1stWon, l_2ndWon,
               l_SvGms, l_bpSaved, l_bpFaced
        FROM matches
        WHERE (winner_name=%s OR loser_name=%s) AND date<=%s
        ORDER BY date DESC LIMIT 50
    """, (player_name, player_name, match_date))
    rows = cursor.fetchall()
    if not rows:
        return

    cols = [
        'date', 'winner_name', 'loser_name', 'surface', 'tourney_level',
        'winner_rank', 'loser_rank', 'winner_rank_points', 'loser_rank_points',
        'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon',
        'w_SvGms', 'w_bpSaved', 'w_bpFaced',
        'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon',
        'l_SvGms', 'l_bpSaved', 'l_bpFaced',
    ]
    df = pd.DataFrame(rows, columns=cols)
    df['is_w'] = df['winner_name'] == player_name

    latest     = df.iloc[0]
    is_w       = bool(latest['is_w'])
    curr_rank  = _safe_int(latest['winner_rank'] if is_w else latest['loser_rank'])
    curr_pts   = _safe_int(latest['winner_rank_points'] if is_w else latest['loser_rank_points'])
    last_date  = latest['date']
    days_since = (date.today() - last_date).days if last_date else 999

    last10     = df.head(10)
    wins10     = int(last10['is_w'].sum())
    matches10  = len(last10)

    cutoff15   = match_date - timedelta(days=15)
    matches15d = int(df[df['date'] >= cutoff15]['date'].count())

    surf       = _safe(latest['surface'], 'Hard')
    surf_rows  = df[df['surface'] == surf].head(20)
    form_s20   = float(surf_rows['is_w'].sum() / len(surf_rows)) if len(surf_rows) else 0.5

    def lvl_form(lvl):
        r = df[df['tourney_level'] == lvl].head(10)
        return float(r['is_w'].sum() / len(r)) if len(r) else 0.5

    def _calc_serve(row):
        s = 'w' if row['is_w'] else 'l'
        o = 'l' if row['is_w'] else 'w'
        svpt = _safe_float(row[f'{s}_svpt']) or 0
        if svpt == 0:
            return None, None
        fi   = _safe_float(row[f'{s}_1stIn'])   or 0
        fw   = _safe_float(row[f'{s}_1stWon'])  or 0
        sw   = _safe_float(row[f'{s}_2ndWon'])  or 0
        bps  = _safe_float(row[f'{s}_bpSaved']) or 0
        bpf  = _safe_float(row[f'{s}_bpFaced']) or 0
        ace  = _safe_float(row[f'{s}_ace'])     or 0
        dfv  = _safe_float(row[f'{s}_df'])      or 0
        s2pt = svpt - fi
        o_svpt = _safe_float(row[f'{o}_svpt'])    or 0
        o_1in  = _safe_float(row[f'{o}_1stIn'])   or 0
        o_1w   = _safe_float(row[f'{o}_1stWon'])  or 0
        o_2w   = _safe_float(row[f'{o}_2ndWon'])  or 0
        o_bps  = _safe_float(row[f'{o}_bpSaved']) or 0
        o_bpf  = _safe_float(row[f'{o}_bpFaced']) or 0
        o_2pt  = o_svpt - o_1in
        def d(a, b): return a/b if b else None
        srv = {
            'first_serve_pct':      d(fi, svpt),
            'first_serve_won_pct':  d(fw, fi),
            'second_serve_won_pct': d(sw, s2pt),
            'bp_save_pct':          d(bps, bpf),
            'ace_rate':             d(ace, svpt),
            'df_rate':              d(dfv, svpt),
        }
        ret = {
            'first_serve_return_won_pct':  d(o_1in - o_1w, o_1in),
            'second_serve_return_won_pct': d(o_2pt - o_2w, o_2pt),
            'bp_conversion_pct':           d(o_bpf - o_bps, o_bpf),
        }
        return srv, ret

    srv = ret = None
    for _, row in df.iterrows():
        srv, ret = _calc_serve(row)
        if srv and all(v is not None for v in srv.values()):
            break
    srv = srv or {}
    ret = ret or {}

    cursor.execute("""
        INSERT INTO player_stats (
            player_id, current_rank, current_rank_points,
            elo_overall, elo_hard, elo_clay, elo_grass, elo_carpet,
            wins_last_10, matches_last_10, matches_last_15d,
            last_match_date, days_since_last_match, form_surface_20,
            form_level_G, form_level_M, form_level_500, form_level_250,
            first_serve_pct, first_serve_won_pct, second_serve_won_pct,
            bp_save_pct, ace_rate, df_rate,
            first_serve_return_won_pct, second_serve_return_won_pct, bp_conversion_pct
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                  %s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON DUPLICATE KEY UPDATE
            current_rank=VALUES(current_rank),
            current_rank_points=VALUES(current_rank_points),
            elo_overall=VALUES(elo_overall), elo_hard=VALUES(elo_hard),
            elo_clay=VALUES(elo_clay), elo_grass=VALUES(elo_grass),
            elo_carpet=VALUES(elo_carpet),
            wins_last_10=VALUES(wins_last_10), matches_last_10=VALUES(matches_last_10),
            matches_last_15d=VALUES(matches_last_15d),
            last_match_date=VALUES(last_match_date),
            days_since_last_match=VALUES(days_since_last_match),
            form_surface_20=VALUES(form_surface_20),
            form_level_G=VALUES(form_level_G), form_level_M=VALUES(form_level_M),
            form_level_500=VALUES(form_level_500), form_level_250=VALUES(form_level_250),
            first_serve_pct=VALUES(first_serve_pct),
            first_serve_won_pct=VALUES(first_serve_won_pct),
            second_serve_won_pct=VALUES(second_serve_won_pct),
            bp_save_pct=VALUES(bp_save_pct), ace_rate=VALUES(ace_rate),
            df_rate=VALUES(df_rate),
            first_serve_return_won_pct=VALUES(first_serve_return_won_pct),
            second_serve_return_won_pct=VALUES(second_serve_return_won_pct),
            bp_conversion_pct=VALUES(bp_conversion_pct)
    """, (
        player_id, curr_rank, curr_pts,
        elos['overall'], elos['Hard'], elos['Clay'], elos['Grass'], elos['Carpet'],
        wins10, matches10, matches15d, last_date, days_since,
        round(form_s20, 4),
        round(lvl_form('G'), 4), round(lvl_form('M'), 4),
        round(lvl_form('500'), 4), round(lvl_form('250'), 4),
        srv.get('first_serve_pct'), srv.get('first_serve_won_pct'),
        srv.get('second_serve_won_pct'), srv.get('bp_save_pct'),
        srv.get('ace_rate'), srv.get('df_rate'),
        ret.get('first_serve_return_won_pct'),
        ret.get('second_serve_return_won_pct'),
        ret.get('bp_conversion_pct'),
    ))

# ── H2H ──────────────────────────────────────────────────────

def _update_h2h(cursor, winner_id, loser_id, surface=None):
    if winner_id < loser_id:
        pa, pb, a_won = winner_id, loser_id, True
    else:
        pa, pb, a_won = loser_id, winner_id, False

    cursor.execute(
        "SELECT h2h_id, total_matches, player_a_wins, player_b_wins, "
        "hard_matches, player_a_wins_hard, clay_matches, player_a_wins_clay, "
        "grass_matches, player_a_wins_grass "
        "FROM h2h_history WHERE player_a_id=%s AND player_b_id=%s", (pa, pb)
    )
    row = cursor.fetchone()
    surf = str(surface).capitalize() if surface else None

    if row:
        h2h_id, total, pa_w, pb_w, hard_m, pa_wh, clay_m, pa_wc, grass_m, pa_wg = row
        total += 1
        if a_won: pa_w += 1
        else:     pb_w += 1

        if surf == 'Hard':
            hard_m  += 1
            if a_won: pa_wh += 1
        elif surf == 'Clay':
            clay_m  += 1
            if a_won: pa_wc += 1
        elif surf == 'Grass':
            grass_m += 1
            if a_won: pa_wg += 1

        cursor.execute(
            "SELECT winner_id FROM matches "
            "WHERE ((winner_id=%s AND loser_id=%s) OR (winner_id=%s AND loser_id=%s)) "
            "ORDER BY date DESC LIMIT 3", (pa, pb, pb, pa)
        )
        l3rows = cursor.fetchall()
        l3    = len(l3rows)
        pa_l3 = sum(1 for r in l3rows if r[0] == pa)
        cursor.execute("""
            UPDATE h2h_history
            SET total_matches=%s, player_a_wins=%s, player_b_wins=%s,
                last_3_matches=%s, player_a_wins_last_3=%s, player_b_wins_last_3=%s,
                hard_matches=%s, player_a_wins_hard=%s,
                clay_matches=%s, player_a_wins_clay=%s,
                grass_matches=%s, player_a_wins_grass=%s
            WHERE h2h_id=%s
        """, (total, pa_w, pb_w, l3, pa_l3, l3 - pa_l3,
              hard_m, pa_wh, clay_m, pa_wc, grass_m, pa_wg,
              h2h_id))
    else:
        pa_w   = 1 if a_won else 0
        hard_m = pa_wh = clay_m = pa_wc = grass_m = pa_wg = 0
        if surf == 'Hard':
            hard_m = 1; pa_wh = 1 if a_won else 0
        elif surf == 'Clay':
            clay_m = 1; pa_wc = 1 if a_won else 0
        elif surf == 'Grass':
            grass_m = 1; pa_wg = 1 if a_won else 0

        cursor.execute("""
            INSERT INTO h2h_history
                (player_a_id, player_b_id, total_matches,
                 player_a_wins, player_b_wins,
                 last_3_matches, player_a_wins_last_3, player_b_wins_last_3,
                 hard_matches, player_a_wins_hard,
                 clay_matches, player_a_wins_clay,
                 grass_matches, player_a_wins_grass)
            VALUES (%s,%s,1,%s,%s,1,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (pa, pb, pa_w, 1 - pa_w, pa_w, 1 - pa_w,
              hard_m, pa_wh, clay_m, pa_wc, grass_m, pa_wg))

# ── MATCH FEATURES ───────────────────────────────────────────

def _insert_match_features(cursor, match_id, r, winner_id, loser_id, elos_w, elos_l, match_date):
    """
    Inserts into match_features using the exact 44-column set in the DB.
    Age is NOT stored here — calculated on the fly from players.birth_date.
    """
    surface  = _safe(r.get('surface'), 'Hard')
    surf_key = surface if surface in ('Hard', 'Clay', 'Grass', 'Carpet') else 'Hard'
    t_level  = _safe(r.get('tourney_level'), '250')

    rank_a = _safe_float(r.get('winner_rank'), 200) or 200.0
    rank_b = _safe_float(r.get('loser_rank'),  200) or 200.0
    pts_a  = _safe_float(r.get('winner_pts'),  0)   or 0.0
    pts_b  = _safe_float(r.get('loser_pts'),   0)   or 0.0
    ht_a   = _safe_float(r.get('winner_ht'),   185) or 185.0
    ht_b   = _safe_float(r.get('loser_ht'),    185) or 185.0

    elo_a_ovr  = elos_w['overall']
    elo_b_ovr  = elos_l['overall']
    elo_a_surf = elos_w.get(surf_key, ELO_DEFAULT)
    elo_b_surf = elos_l.get(surf_key, ELO_DEFAULT)

    wname    = str(r.get('winner_name', ''))
    lname    = str(r.get('loser_name',  ''))
    cutoff15 = match_date - timedelta(days=15)

    def pform(pname):
        cursor.execute("""
            SELECT winner_name, surface, tourney_level FROM matches
            WHERE (winner_name=%s OR loser_name=%s) AND date<%s
            ORDER BY date DESC LIMIT 50
        """, (pname, pname, match_date))
        rows = cursor.fetchall()
        if not rows: return 0.5, 0.5, 0.5
        def wr(rs): return float(sum(1 for x in rs if x[0]==pname)/len(rs)) if rs else 0.5
        return (wr(rows[:10]),
                wr([x for x in rows if x[1]==surface][:20]),
                wr([x for x in rows if x[2]==t_level][:10]))

    def fatigue(pname):
        cursor.execute(
            "SELECT COUNT(*) FROM matches "
            "WHERE (winner_name=%s OR loser_name=%s) AND date>=%s AND date<%s",
            (pname, pname, cutoff15, match_date)
        )
        return cursor.fetchone()[0]

    def inactive(pname):
        cursor.execute(
            "SELECT MAX(date) FROM matches "
            "WHERE (winner_name=%s OR loser_name=%s) AND date<%s",
            (pname, pname, match_date)
        )
        last = cursor.fetchone()[0]
        return 1 if (not last or (match_date - last).days > 30) else 0

    a_ovr, a_s20, a_lvl = pform(wname)
    b_ovr, b_s20, b_lvl = pform(lname)
    fat_a = fatigue(wname); fat_b = fatigue(lname)
    ina_a = inactive(wname); ina_b = inactive(lname)

    pa_id, pb_id = (winner_id, loser_id) if winner_id < loser_id else (loser_id, winner_id)
    cursor.execute("""
        SELECT total_matches, player_a_wins, player_b_wins,
               last_3_matches, player_a_wins_last_3, player_b_wins_last_3
        FROM h2h_history WHERE player_a_id=%s AND player_b_id=%s
    """, (pa_id, pb_id))
    h2h = cursor.fetchone()
    if h2h:
        tot_h2h, pa_w, pb_w, l3, pa_l3, pb_l3 = h2h
        if pa_id == winner_id: a_h2h, b_h2h, a_l3, b_l3 = pa_w, pb_w, pa_l3, pb_l3
        else:                  a_h2h, b_h2h, a_l3, b_l3 = pb_w, pa_w, pb_l3, pa_l3
    else:
        tot_h2h = a_h2h = b_h2h = l3 = a_l3 = b_l3 = 0

    # Serve/return diffs from match row
    def sp(c): return _safe_float(r.get(c)) or 0
    w_svpt=sp('w_svpt'); l_svpt=sp('l_svpt')
    w_1in=sp('w_1stIn'); l_1in=sp('l_1stIn')
    w_1w=sp('w_1stWon'); l_1w=sp('l_1stWon')
    w_2w=sp('w_2ndWon'); l_2w=sp('l_2ndWon')
    w_bps=sp('w_bpSaved'); l_bps=sp('l_bpSaved')
    w_bpf=sp('w_bpFaced'); l_bpf=sp('l_bpFaced')
    w_2pt=w_svpt-w_1in; l_2pt=l_svpt-l_1in

    def d(a, b): return a/b if b else None
    def oz(v):   return v if v is not None else 0

    fsr_a=d(l_1in-l_1w, l_1in) if l_1in  else None
    fsr_b=d(w_1in-w_1w, w_1in) if w_1in  else None
    ssr_a=d(l_2pt-l_2w, l_2pt) if l_2pt  else None
    ssr_b=d(w_2pt-w_2w, w_2pt) if w_2pt  else None
    bpc_a=d(l_bpf-l_bps, l_bpf) if l_bpf else None
    bpc_b=d(w_bpf-w_bps, w_bpf) if w_bpf else None

    # player_a is always winner; player_a_won = 1
    vals = (
        match_id,
        winner_id, loser_id,                          # player_a_id, player_b_id
        elo_a_ovr, elo_b_ovr,                         # elo_overall
        elo_a_ovr - elo_b_ovr,                        # elo_diff_overall
        elo_a_surf, elo_b_surf,                       # elo_surface
        elo_a_surf - elo_b_surf,                      # elo_diff_surface
        rank_a, rank_b,                               # rank
        rank_a - rank_b,                              # rank_diff
        pts_a, pts_b,                                 # points
        pts_a - pts_b,                                # points_diff
        ht_a, ht_b,                                   # height
        ht_a - ht_b,                                  # height_diff
        a_ovr, b_ovr,                                 # form_overall
        a_s20, b_s20,                                 # form_surface
        ina_a, ina_b,                                 # inactive flags
        tot_h2h, a_h2h, b_h2h,                       # h2h totals
        l3, a_l3, b_l3,                               # last3 h2h
        fat_a, fat_b,                                 # matches_last15d
        a_l3 - b_l3,                                  # last3_h2h_diff
        fat_a - fat_b,                                # matches_last15d_diff
        1,                                            # player_a_won
        a_s20 - b_s20,                                # form_surface_20_diff
        a_lvl, b_lvl,                                 # form_level
        a_lvl - b_lvl,                                # form_level_diff
        oz(fsr_a) - oz(fsr_b),                        # first_serve_return_won_pct_diff
        oz(ssr_a) - oz(ssr_b),                        # second_serve_return_won_pct_diff
        oz(bpc_a) - oz(bpc_b),                        # bp_conversion_pct_diff
        a_s20, b_s20,                                 # form_surface_20 (explicit columns)
    )

    cursor.execute("""
        INSERT INTO match_features (
            match_id,
            player_a_id, player_b_id,
            player_a_elo_overall, player_b_elo_overall, elo_diff_overall,
            player_a_elo_surface, player_b_elo_surface, elo_diff_surface,
            player_a_rank, player_b_rank, rank_diff,
            player_a_points, player_b_points, points_diff,
            player_a_height, player_b_height, height_diff,
            player_a_form_overall, player_b_form_overall,
            player_a_form_surface, player_b_form_surface,
            player_a_inactive, player_b_inactive,
            total_h2h, player_a_h2h_wins, player_b_h2h_wins,
            total_last3_h2h, player_a_last3_h2h_wins, player_b_last3_h2h_wins,
            player_a_matches_last15d, player_b_matches_last15d,
            last3_h2h_diff, matches_last15d_diff,
            player_a_won,
            form_surface_20_diff,
            player_a_form_level, player_b_form_level, form_level_diff,
            first_serve_return_won_pct_diff,
            second_serve_return_won_pct_diff,
            bp_conversion_pct_diff,
            player_a_form_surface_20, player_b_form_surface_20
        ) VALUES (
            %s,
            %s,%s,
            %s,%s,%s,
            %s,%s,%s,
            %s,%s,%s,
            %s,%s,%s,
            %s,%s,%s,
            %s,%s,
            %s,%s,
            %s,%s,
            %s,%s,%s,
            %s,%s,%s,
            %s,%s,
            %s,%s,
            %s,
            %s,
            %s,%s,%s,
            %s,
            %s,
            %s,
            %s,%s
        ) ON DUPLICATE KEY UPDATE match_id=match_id
    """, vals)

# ── MAIN ─────────────────────────────────────────────────────

def run_update(year=None):
    if year is None:
        year = datetime.now().year
    print(f"[updater] Starting update — year={year}")

    index = _fetch_index()
    if not index:
        print("[updater] No file index. Aborting.")
        return False

    frames = []
    year_file = f"{year}.csv"
    if year_file in index:
        df = _download_csv(index[year_file], year_file)
        if df is not None: frames.append(df)
    else:
        print(f"[updater] {year_file} not found in index.")

    if ONGOING_FILE in index:
        df = _download_csv(index[ONGOING_FILE], ONGOING_FILE)
        if df is not None:
            df['_source'] = 'ongoing'
            frames.append(df)
    else:
        print(f"[updater] {ONGOING_FILE} not in index.")

    if not frames:
        print("[updater] Nothing downloaded. Aborting.")
        return False

    df_all = pd.concat(frames, ignore_index=True)
    df_all = _normalise(df_all)
    df_all = df_all.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
    print(f"[updater] {len(df_all)} rows after normalise")

    conn = get_db_connection()
    if not conn:
        print("[updater] Cannot connect to DB. Aborting.")
        return False
    cursor = conn.cursor(buffered=True)

    latest_db = _latest_date_in_db(cursor)
    since     = latest_db - timedelta(days=30)
    existing  = _existing_keys(cursor, since)
    print(f"[updater] Latest in DB: {latest_db}. Look-back from {since}.")

    df_new = df_all[df_all['date'] > latest_db].copy()
    df_ovl = df_all[(df_all['date'] >= since) & (df_all['date'] <= latest_db)].copy()
    df_ovl = df_ovl[~df_ovl.apply(
        lambda row: (row['date'], row['winner_name'], row['loser_name']) in existing, axis=1
    )]

    df_ins = pd.concat([df_ovl, df_new], ignore_index=True)
    df_ins = df_ins.sort_values('date').drop_duplicates(
        subset=['date', 'winner_name', 'loser_name']
    ).reset_index(drop=True)

    if df_ins.empty:
        print(f"[updater] Already up to date (latest: {latest_db}).")
        conn.close()
        return False

    print(f"[updater] Inserting {len(df_ins)} new matches …")
    inserted = 0

    for _, row in df_ins.iterrows():
        try:
            r          = row.to_dict()
            match_date = r['date']
            wname      = str(r['winner_name']).strip()
            lname      = str(r['loser_name']).strip()
            surface    = _safe(r.get('surface'), 'Hard')
            t_level    = _safe(r.get('tourney_level'), '250')
            surf_key   = surface if surface in ('Hard','Clay','Grass','Carpet') else 'Hard'
            indoor_val = _safe(r.get('indoor'), 'O')

            # Estimate birth_date from TML age float — COALESCE ensures existing
            # players with a known birth_date (from backfill) are never overwritten
            w_bd = _age_to_birthdate(_safe_float(r.get('winner_age')), match_date)
            l_bd = _age_to_birthdate(_safe_float(r.get('loser_age')),  match_date)

            w_id = _get_or_create_player(cursor, wname,
                hand=_safe(r.get('winner_hand')), ht=r.get('winner_ht'),
                nationality=_safe(r.get('winner_ioc')), atp_id=r.get('winner_atp_id'),
                birth_date=w_bd)
            l_id = _get_or_create_player(cursor, lname,
                hand=_safe(r.get('loser_hand')), ht=r.get('loser_ht'),
                nationality=_safe(r.get('loser_ioc')), atp_id=r.get('loser_atp_id'),
                birth_date=l_bd)

            elos_w = _get_elos(cursor, w_id)
            elos_l = _get_elos(cursor, l_id)

            cursor.execute("""
                INSERT IGNORE INTO matches (
                    date, tournament, surface, round, tourney_level,
                    indoor, best_of, winner_id, winner_atp_id, loser_id, loser_atp_id,
                    winner_name, loser_name,
                    winner_rank, loser_rank, winner_rank_points, loser_rank_points,
                    score, minutes,
                    w_ace, w_df, w_svpt, w_1stIn, w_1stWon, w_2ndWon,
                    w_SvGms, w_bpSaved, w_bpFaced,
                    l_ace, l_df, l_svpt, l_1stIn, l_1stWon, l_2ndWon,
                    l_SvGms, l_bpSaved, l_bpFaced
                ) VALUES (
                    %s,%s,%s,%s,%s,
                    %s,%s,%s,%s,%s,%s,
                    %s,%s,
                    %s,%s,%s,%s,
                    %s,%s,
                    %s,%s,%s,%s,%s,%s,%s,%s,%s,
                    %s,%s,%s,%s,%s,%s,%s,%s,%s
                )
            """, (
                match_date, _safe(r.get('tournament')), surface,
                _safe(r.get('round')), t_level,
                indoor_val, _safe_int(r.get('best_of'), 3),
                w_id, _safe(r.get('winner_atp_id')),
                l_id, _safe(r.get('loser_atp_id')),
                wname, lname,
                _safe_int(r.get('winner_rank')),  _safe_int(r.get('loser_rank')),
                _safe_int(r.get('winner_pts')),   _safe_int(r.get('loser_pts')),
                _safe(r.get('score')), _safe_int(r.get('minutes')),
                _safe_int(r.get('w_ace')),    _safe_int(r.get('w_df')),
                _safe_int(r.get('w_svpt')),   _safe_int(r.get('w_1stIn')),
                _safe_int(r.get('w_1stWon')), _safe_int(r.get('w_2ndWon')),
                _safe_int(r.get('w_SvGms')),  _safe_int(r.get('w_bpSaved')),
                _safe_int(r.get('w_bpFaced')),
                _safe_int(r.get('l_ace')),    _safe_int(r.get('l_df')),
                _safe_int(r.get('l_svpt')),   _safe_int(r.get('l_1stIn')),
                _safe_int(r.get('l_1stWon')), _safe_int(r.get('l_2ndWon')),
                _safe_int(r.get('l_SvGms')),  _safe_int(r.get('l_bpSaved')),
                _safe_int(r.get('l_bpFaced')),
            ))

            if cursor.rowcount == 0:
                continue  # duplicate — already in DB

            match_id = cursor.lastrowid
            K = _k_factor(t_level)
            new_w_ovr, new_l_ovr   = _elo_update(elos_w['overall'],  elos_l['overall'],  K)
            new_w_surf, new_l_surf = _elo_update(elos_w[surf_key],   elos_l[surf_key],   K)

            elos_w_new = dict(elos_w); elos_l_new = dict(elos_l)
            elos_w_new['overall'] = new_w_ovr;  elos_l_new['overall'] = new_l_ovr
            elos_w_new[surf_key]  = new_w_surf; elos_l_new[surf_key]  = new_l_surf

            _save_elo(cursor, w_id, elos_w_new, match_date)
            _save_elo(cursor, l_id, elos_l_new, match_date)
            _recalc_player_stats(cursor, w_id, wname, match_date, elos_w_new)
            _recalc_player_stats(cursor, l_id, lname, match_date, elos_l_new)
            _update_h2h(cursor, w_id, l_id, surface=surface)
            _insert_match_features(cursor, match_id, r, w_id, l_id, elos_w, elos_l, match_date)

            conn.commit()
            inserted += 1
            if inserted % 50 == 0:
                print(f"[updater]   … {inserted} inserted so far")

        except Exception as e:
            conn.rollback()
            print(f"[updater] Error on {row.get('date')} "
                  f"{row.get('winner_name')} vs {row.get('loser_name')}: {e}")
            continue

    conn.close()
    print(f"[updater] Done. {inserted} new matches inserted.")
    return inserted > 0


if __name__ == "__main__":
    run_update()
