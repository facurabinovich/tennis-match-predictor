"""
Initial Data Load Script - TML Dataset
Populates MySQL database with historical match data from df_neutral
Run ONCE after truncating all tables
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
from config_local import get_db_connection

print("="*60)
print("INITIAL DATA LOAD - TML DATASET")
print("="*60)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading data from pickle files...")
df_neutral = pd.read_pickle('db_migration/matches_neutral.pkl')
print(f"✓ Loaded matches_neutral.pkl: {len(df_neutral):,} matches")

with open('db_migration/player_elo.pkl', 'rb') as f:
    player_elo_v2 = pickle.load(f)
print(f"✓ Loaded player_elo.pkl: {len(player_elo_v2):,} players")

# Load raw df for match stats (score, minutes, w_ace, etc.)
df_raw = pd.read_pickle('db_migration/df_raw.pkl')
df_raw['tourney_date'] = pd.to_datetime(df_raw['tourney_date'].astype(str), format='%Y%m%d')
print(f"✓ Loaded df_raw.pkl: {len(df_raw):,} matches")

# Build stats lookup: (date, winner_name, loser_name) -> stats
stats_lookup = {}
for _, row in df_raw.iterrows():
    key = (row['tourney_date'].date(), row['winner_name'], row['loser_name'])
    stats_lookup[key] = {
        'score':     row.get('score'),
        'minutes':   row.get('minutes'),
        'w_ace':     row.get('w_ace'),     'l_ace':     row.get('l_ace'),
        'w_df':      row.get('w_df'),      'l_df':      row.get('l_df'),
        'w_svpt':    row.get('w_svpt'),    'l_svpt':    row.get('l_svpt'),
        'w_1stIn':   row.get('w_1stIn'),   'l_1stIn':   row.get('l_1stIn'),
        'w_1stWon':  row.get('w_1stWon'),  'l_1stWon':  row.get('l_1stWon'),
        'w_2ndWon':  row.get('w_2ndWon'),  'l_2ndWon':  row.get('l_2ndWon'),
        'w_SvGms':   row.get('w_SvGms'),   'l_SvGms':   row.get('l_SvGms'),
        'w_bpSaved': row.get('w_bpSaved'), 'l_bpSaved': row.get('l_bpSaved'),
        'w_bpFaced': row.get('w_bpFaced'), 'l_bpFaced': row.get('l_bpFaced'),
    }
print(f"✓ Built stats lookup: {len(stats_lookup):,} entries")

connection = get_db_connection()
if not connection:
    print("✗ Failed to connect to database")
    exit(1)

cursor = connection.cursor()
print("✓ Connected to MySQL database")

df_sorted = df_neutral.sort_values('tourney_date').reset_index(drop=True)

def safe_int(v):
    return int(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else None

def safe_float(v):
    return float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else None

def safe_str(v):
    return str(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else None

def parse_date(v):
    return datetime.strptime(str(int(v)), '%Y%m%d').date()

# ============================================================
# STEP 1: POPULATE PLAYERS
# ============================================================

print("\n" + "="*60)
print("STEP 1: POPULATING PLAYERS")
print("="*60)

all_players = {}
for _, row in df_neutral.iterrows():
    for side in ['player_a', 'player_b']:
        name = row[f'{side}_name']
        if name not in all_players:
            all_players[name] = {
                'atp_id':      safe_str(row[f'{side}_id']),
                'hand':        safe_str(row[f'{side}_hand']),
                'height':      safe_int(row[f'{side}_height']),
                'nationality': None,
            }

# Enrich with nationality from df_raw (winner_ioc / loser_ioc)
print("Enriching players with nationality from raw data...")
for _, row in df_raw.iterrows():
    w = row['winner_name']
    l = row['loser_name']
    if w in all_players and all_players[w]['nationality'] is None:
        all_players[w]['nationality'] = safe_str(row.get('winner_ioc'))
    if l in all_players and all_players[l]['nationality'] is None:
        all_players[l]['nationality'] = safe_str(row.get('loser_ioc'))

covered = sum(1 for p in all_players.values() if p['nationality'] is not None)
print(f"Nationality coverage: {covered:,}/{len(all_players):,} players")

print(f"Inserting {len(all_players):,} players...")
inserted = skipped = 0

for name, info in all_players.items():
    try:
        cursor.execute("""
            INSERT IGNORE INTO players (player_name, hand, height, atp_id, nationality)
            VALUES (%s, %s, %s, %s, %s)
        """, (name, info['hand'], info['height'], info['atp_id'], info['nationality']))
        if cursor.rowcount > 0:
            inserted += 1
        else:
            skipped += 1
    except Exception as e:
        print(f"  Error inserting {name}: {e}")

connection.commit()
print(f"✓ Inserted {inserted:,} players, skipped {skipped} duplicates")

# Build player_id lookup
cursor.execute("SELECT player_id, player_name FROM players")
player_ids = {name: pid for pid, name in cursor.fetchall()}
print(f"  Player ID map: {len(player_ids):,} entries")

# ============================================================
# STEP 2: POPULATE MATCHES
# ============================================================

print("\n" + "="*60)
print("STEP 2: POPULATING MATCHES")
print("="*60)

inserted = errors = 0
batch = []

for _, row in df_sorted.iterrows():
    try:
        a_name = row['player_a_name']
        b_name = row['player_b_name']

        if a_name not in player_ids or b_name not in player_ids:
            errors += 1
            continue

        a_id = player_ids[a_name]
        b_id = player_ids[b_name]
        a_won = row['target'] == 1

        w = 'player_a' if a_won else 'player_b'
        l = 'player_b' if a_won else 'player_a'

        winner_id   = a_id   if a_won else b_id
        winner_name = a_name if a_won else b_name
        loser_id    = b_id   if a_won else a_id
        loser_name  = b_name if a_won else a_name

        match_date = parse_date(row['tourney_date'])
        stats = stats_lookup.get((match_date, winner_name, loser_name), {})

        batch.append((
            match_date,
            safe_str(row.get('tourney_name')),
            safe_str(row.get('tourney_id')),
            safe_str(row.get('tourney_level')),
            safe_str(row.get('indoor')),
            safe_str(row.get('surface')),
            safe_str(row.get('round')),
            safe_int(row.get('best_of')),
            winner_id, safe_str(row.get(f'{w}_id')), winner_name,
            safe_int(row.get(f'{w}_rank')), safe_int(row.get(f'{w}_points')),
            loser_id,  safe_str(row.get(f'{l}_id')), loser_name,
            safe_int(row.get(f'{l}_rank')), safe_int(row.get(f'{l}_points')),
            safe_str(stats.get('score')),
            safe_int(stats.get('minutes')),
            safe_int(stats.get('w_ace')),     safe_int(stats.get('l_ace')),
            safe_int(stats.get('w_df')),      safe_int(stats.get('l_df')),
            safe_int(stats.get('w_svpt')),    safe_int(stats.get('l_svpt')),
            safe_int(stats.get('w_1stIn')),   safe_int(stats.get('l_1stIn')),
            safe_int(stats.get('w_1stWon')),  safe_int(stats.get('l_1stWon')),
            safe_int(stats.get('w_2ndWon')),  safe_int(stats.get('l_2ndWon')),
            safe_int(stats.get('w_SvGms')),   safe_int(stats.get('l_SvGms')),
            safe_int(stats.get('w_bpSaved')), safe_int(stats.get('l_bpSaved')),
            safe_int(stats.get('w_bpFaced')), safe_int(stats.get('l_bpFaced')),
        ))
        inserted += 1

        if len(batch) >= 5000:
            cursor.executemany("""
                INSERT INTO matches (
                    date, tournament, tournament_id, tourney_level, indoor,
                    surface, round, best_of,
                    winner_id, winner_atp_id, winner_name, winner_rank, winner_rank_points,
                    loser_id,  loser_atp_id,  loser_name,  loser_rank,  loser_rank_points,
                    score, minutes,
                    w_ace, l_ace, w_df, l_df, w_svpt, l_svpt,
                    w_1stIn, l_1stIn, w_1stWon, l_1stWon, w_2ndWon, l_2ndWon,
                    w_SvGms, l_SvGms, w_bpSaved, l_bpSaved, w_bpFaced, l_bpFaced
                ) VALUES (
                    %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                    %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s
                )
            """, batch)
            connection.commit()
            batch = []
            print(f"  Inserted {inserted:,} matches...")

    except Exception as e:
        errors += 1
        if errors <= 5:
            print(f"  Error: {e}")

if batch:
    cursor.executemany("""
        INSERT INTO matches (
            date, tournament, tournament_id, tourney_level, indoor,
            surface, round, best_of,
            winner_id, winner_atp_id, winner_name, winner_rank, winner_rank_points,
            loser_id,  loser_atp_id,  loser_name,  loser_rank,  loser_rank_points,
            score, minutes,
            w_ace, l_ace, w_df, l_df, w_svpt, l_svpt,
            w_1stIn, l_1stIn, w_1stWon, l_1stWon, w_2ndWon, l_2ndWon,
            w_SvGms, l_SvGms, w_bpSaved, l_bpSaved, w_bpFaced, l_bpFaced
        ) VALUES (
            %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
            %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s
        )
    """, batch)
    connection.commit()

print(f"✓ Inserted {inserted:,} matches, {errors} errors")

# Build match lookup
cursor.execute("SELECT match_id, date, winner_name, loser_name FROM matches")
match_lookup = {(str(r[1]), r[2], r[3]): r[0] for r in cursor.fetchall()}
print(f"  Match lookup: {len(match_lookup):,} entries")

# ============================================================
# STEP 3: POPULATE PLAYER_ELO
# ============================================================

print("\n" + "="*60)
print("STEP 3: POPULATING PLAYER_ELO")
print("="*60)

elo_records = []
for name, pid in player_ids.items():
    mask_a = df_sorted['player_a_name'] == name
    mask_b = df_sorted['player_b_name'] == name
    pm = df_sorted[mask_a | mask_b]
    if len(pm) == 0:
        continue

    last  = pm.iloc[-1]
    side  = 'player_a' if last['player_a_name'] == name else 'player_b'
    match_date = parse_date(last['tourney_date'])
    surface    = last['surface']

    elo_overall = safe_float(last[f'{side}_elo_overall']) or 1500.0
    elo_surface = safe_float(last[f'{side}_elo_surface']) or 1500.0

    elo_records.append((
        pid, match_date, elo_overall,
        elo_surface if surface == 'Hard'   else 1500.0,
        elo_surface if surface == 'Clay'   else 1500.0,
        elo_surface if surface == 'Grass'  else 1500.0,
        elo_surface if surface == 'Carpet' else 1500.0,
    ))

cursor.executemany("""
    INSERT INTO player_elo (player_id, date, elo_overall, elo_hard, elo_clay, elo_grass, elo_carpet)
    VALUES (%s,%s,%s,%s,%s,%s,%s)
    ON DUPLICATE KEY UPDATE
        elo_overall = VALUES(elo_overall),
        elo_hard    = VALUES(elo_hard),
        elo_clay    = VALUES(elo_clay),
        elo_grass   = VALUES(elo_grass),
        elo_carpet  = VALUES(elo_carpet)
""", elo_records)
connection.commit()
print(f"✓ Inserted {len(elo_records):,} Elo records")

# ============================================================
# STEP 4: POPULATE PLAYER_STATS
# ============================================================

print("\n" + "="*60)
print("STEP 4: POPULATING PLAYER_STATS")
print("="*60)

updated = errors_ps = 0

for name, pid in player_ids.items():
    try:
        mask_a = df_sorted['player_a_name'] == name
        mask_b = df_sorted['player_b_name'] == name
        pm = df_sorted[mask_a | mask_b]
        if len(pm) == 0:
            continue

        last  = pm.iloc[-1]
        side  = 'player_a' if last['player_a_name'] == name else 'player_b'
        last_date  = parse_date(last['tourney_date'])
        days_since = (datetime.now().date() - last_date).days

        # Form last 10
        recent10 = pm.tail(10)
        wins10 = sum(
            1 for _, m in recent10.iterrows()
            if (m['player_a_name'] == name and m['target'] == 1) or
               (m['player_b_name'] == name and m['target'] == 0)
        )

        # Fatigue last 15d
        cutoff = datetime.combine(last_date, datetime.min.time()) - timedelta(days=15)
        recent15d = sum(
            1 for _, m in pm.iterrows()
            if datetime.strptime(str(int(m['tourney_date'])), '%Y%m%d') >= cutoff
        )

        surface = last['surface']

        cursor.execute("""
            INSERT INTO player_stats (
                player_id, current_rank, current_rank_points, age, height, hand,
                elo_overall, elo_hard, elo_clay, elo_grass, elo_carpet,
                wins_last_10, matches_last_10, matches_last_15d,
                last_match_date, days_since_last_match,
                form_surface_20,
                form_level_G, form_level_M, form_level_500, form_level_250,
                first_serve_pct, first_serve_won_pct, second_serve_won_pct,
                bp_save_pct, ace_rate, df_rate,
                first_serve_return_won_pct, second_serve_return_won_pct, bp_conversion_pct
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                      %s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON DUPLICATE KEY UPDATE
                current_rank                 = VALUES(current_rank),
                current_rank_points          = VALUES(current_rank_points),
                age                          = VALUES(age),
                elo_overall                  = VALUES(elo_overall),
                elo_hard                     = VALUES(elo_hard),
                elo_clay                     = VALUES(elo_clay),
                elo_grass                    = VALUES(elo_grass),
                elo_carpet                   = VALUES(elo_carpet),
                wins_last_10                 = VALUES(wins_last_10),
                matches_last_10              = VALUES(matches_last_10),
                matches_last_15d             = VALUES(matches_last_15d),
                last_match_date              = VALUES(last_match_date),
                days_since_last_match        = VALUES(days_since_last_match),
                form_surface_20              = VALUES(form_surface_20),
                form_level_G                 = VALUES(form_level_G),
                form_level_M                 = VALUES(form_level_M),
                form_level_500               = VALUES(form_level_500),
                form_level_250               = VALUES(form_level_250),
                first_serve_pct              = VALUES(first_serve_pct),
                first_serve_won_pct          = VALUES(first_serve_won_pct),
                second_serve_won_pct         = VALUES(second_serve_won_pct),
                bp_save_pct                  = VALUES(bp_save_pct),
                ace_rate                     = VALUES(ace_rate),
                df_rate                      = VALUES(df_rate),
                first_serve_return_won_pct   = VALUES(first_serve_return_won_pct),
                second_serve_return_won_pct  = VALUES(second_serve_return_won_pct),
                bp_conversion_pct            = VALUES(bp_conversion_pct)
        """, (
            pid,
            safe_int(last[f'{side}_rank']),
            safe_int(last[f'{side}_points']),
            safe_float(last[f'{side}_age']),
            safe_int(last[f'{side}_height']),
            safe_str(last[f'{side}_hand']),
            safe_float(last[f'{side}_elo_overall']),
            safe_float(last[f'{side}_elo_surface']) if surface == 'Hard'   else 1500.0,
            safe_float(last[f'{side}_elo_surface']) if surface == 'Clay'   else 1500.0,
            safe_float(last[f'{side}_elo_surface']) if surface == 'Grass'  else 1500.0,
            1500.0,
            wins10,
            len(recent10),
            recent15d,
            last_date,
            days_since,
            safe_float(last[f'{side}_form_surface_20']),
            safe_float(last[f'{side}_form_level']) if last['tourney_level'] == 'G'   else None,
            safe_float(last[f'{side}_form_level']) if last['tourney_level'] == 'M'   else None,
            safe_float(last[f'{side}_form_level']) if last['tourney_level'] == '500' else None,
            safe_float(last[f'{side}_form_level']) if last['tourney_level'] == '250' else None,
            safe_float(last[f'{side}_first_serve_pct']),
            safe_float(last[f'{side}_first_serve_won_pct']),
            safe_float(last[f'{side}_second_serve_won_pct']),
            safe_float(last[f'{side}_bp_save_pct']),
            safe_float(last[f'{side}_ace_rate']),
            safe_float(last[f'{side}_df_rate']),
            safe_float(last[f'{side}_first_serve_return_won_pct']),
            safe_float(last[f'{side}_second_serve_return_won_pct']),
            safe_float(last[f'{side}_bp_conversion_pct']),
        ))
        updated += 1

    except Exception as e:
        errors_ps += 1
        if errors_ps <= 5:
            print(f"  Error {name}: {e}")

connection.commit()
print(f"✓ Updated {updated:,} player stats, {errors_ps} errors")

# ============================================================
# STEP 5: POPULATE MATCH_FEATURES
# ============================================================

print("\n" + "="*60)
print("STEP 5: POPULATING MATCH_FEATURES")
print("="*60)

inserted_mf = errors_mf = 0
batch_mf = []

for _, row in df_sorted.iterrows():
    try:
        a_won       = row['target'] == 1
        a_name      = row['player_a_name']
        b_name      = row['player_b_name']
        winner_name = a_name if a_won else b_name
        loser_name  = b_name if a_won else a_name
        match_date  = parse_date(row['tourney_date'])

        match_id = match_lookup.get((str(match_date), winner_name, loser_name))
        if not match_id:
            errors_mf += 1
            continue

        a_id = player_ids.get(a_name)
        b_id = player_ids.get(b_name)
        if not a_id or not b_id:
            errors_mf += 1
            continue

        batch_mf.append((
            match_id, a_id, b_id,
            # Elo
            safe_float(row['player_a_elo_overall']),
            safe_float(row['player_b_elo_overall']),
            safe_float(row['elo_diff_overall']),
            safe_float(row['player_a_elo_surface']),
            safe_float(row['player_b_elo_surface']),
            safe_float(row['elo_diff_surface']),
            # Rank/points
            safe_int(row['player_a_rank']),
            safe_int(row['player_b_rank']),
            safe_int(row['rank_diff']),
            safe_int(row['player_a_points']),
            safe_int(row['player_b_points']),
            safe_int(row['points_diff']),
            # Physical
            safe_float(row['player_a_age']),
            safe_float(row['player_b_age']),
            safe_float(row['age_diff']),
            safe_int(row['player_a_height']),
            safe_int(row['player_b_height']),
            safe_int(row['height_diff']),
            # Form
            safe_float(row['player_a_form_overall']),
            safe_float(row['player_b_form_overall']),
            safe_float(row['player_a_form_surface']),
            safe_float(row['player_b_form_surface']),
            # Inactivity
            safe_int(row['player_a_inactive']),
            safe_int(row['player_b_inactive']),
            # H2H
            safe_int(row['total_h2h']),
            safe_int(row['player_a_h2h_wins']),
            safe_int(row['player_b_h2h_wins']),
            safe_int(row['total_last3_h2h']),
            safe_int(row['player_a_last3_h2h_wins']),
            safe_int(row['player_b_last3_h2h_wins']),
            # Fatigue
            safe_int(row['player_a_matches_last15d']),
            safe_int(row['player_b_matches_last15d']),
            safe_int(row['last3_h2h_diff']),
            safe_int(row['matches_last15d_diff']),
            # Target
            int(a_won),
            # New enriched features
            safe_float(row['form_surface_20_diff']),
            safe_float(row['player_a_form_level']),
            safe_float(row['player_b_form_level']),
            safe_float(row['form_level_diff']),
            safe_float(row['first_serve_return_won_pct_diff']),
            safe_float(row['second_serve_return_won_pct_diff']),
            safe_float(row['bp_conversion_pct_diff']),
            safe_float(row['player_a_form_surface_20']),
            safe_float(row['player_b_form_surface_20']),
        ))

        inserted_mf += 1

        if len(batch_mf) >= 5000:
            cursor.executemany("""
                INSERT IGNORE INTO match_features (
                    match_id, player_a_id, player_b_id,
                    player_a_elo_overall, player_b_elo_overall, elo_diff_overall,
                    player_a_elo_surface, player_b_elo_surface, elo_diff_surface,
                    player_a_rank, player_b_rank, rank_diff,
                    player_a_points, player_b_points, points_diff,
                    player_a_age, player_b_age, age_diff,
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
                    %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                    %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                    %s,%s,%s,%s,%s
                )
            """, batch_mf)
            connection.commit()
            batch_mf = []
            print(f"  Inserted {inserted_mf:,} match features...")

    except Exception as e:
        errors_mf += 1
        if errors_mf <= 5:
            print(f"  Error: {e}")

if batch_mf:
    cursor.executemany("""
        INSERT IGNORE INTO match_features (
            match_id, player_a_id, player_b_id,
            player_a_elo_overall, player_b_elo_overall, elo_diff_overall,
            player_a_elo_surface, player_b_elo_surface, elo_diff_surface,
            player_a_rank, player_b_rank, rank_diff,
            player_a_points, player_b_points, points_diff,
            player_a_age, player_b_age, age_diff,
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
            %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
            %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
            %s,%s,%s,%s,%s
        )
    """, batch_mf)
    connection.commit()

print(f"✓ Inserted {inserted_mf:,} match features, {errors_mf} errors")

# ============================================================
# STEP 6: POPULATE H2H_HISTORY
# ============================================================

print("\n" + "="*60)
print("STEP 6: POPULATING H2H_HISTORY")
print("="*60)

h2h_dict = {}
for _, row in df_sorted.iterrows():
    a_name = row['player_a_name']
    b_name = row['player_b_name']
    if a_name not in player_ids or b_name not in player_ids:
        continue

    a_id = player_ids[a_name]
    b_id = player_ids[b_name]
    key  = (min(a_id, b_id), max(a_id, b_id))

    h2h_dict[key] = {
        'total':  safe_int(row['total_h2h'])              or 0,
        'a_wins': safe_int(row['player_a_h2h_wins'])      or 0,
        'b_wins': safe_int(row['player_b_h2h_wins'])      or 0,
        'last3':  safe_int(row['total_last3_h2h'])        or 0,
        'a_last3':safe_int(row['player_a_last3_h2h_wins'])or 0,
        'b_last3':safe_int(row['player_b_last3_h2h_wins'])or 0,
    }

h2h_records = [
    (k[0], k[1], v['total'], v['a_wins'], v['b_wins'],
     v['last3'], v['a_last3'], v['b_last3'])
    for k, v in h2h_dict.items()
]

cursor.executemany("""
    INSERT INTO h2h_history (
        player_a_id, player_b_id,
        total_matches, player_a_wins, player_b_wins,
        last_3_matches, player_a_wins_last_3, player_b_wins_last_3
    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
    ON DUPLICATE KEY UPDATE
        total_matches        = VALUES(total_matches),
        player_a_wins        = VALUES(player_a_wins),
        player_b_wins        = VALUES(player_b_wins),
        last_3_matches       = VALUES(last_3_matches),
        player_a_wins_last_3 = VALUES(player_a_wins_last_3),
        player_b_wins_last_3 = VALUES(player_b_wins_last_3)
""", h2h_records)
connection.commit()
print(f"✓ Inserted {len(h2h_records):,} H2H records")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "="*60)
print("✓ INITIAL DATA LOAD COMPLETE")
print("="*60)

for table in ['players', 'matches', 'player_elo', 'player_stats', 'h2h_history', 'match_features']:
    cursor.execute(f"SELECT COUNT(*) FROM {table}")
    print(f"  {table:<20} {cursor.fetchone()[0]:>8,} rows")

cursor.close()
connection.close()
print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
