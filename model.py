# model.py — Wins-first CFB Top-25 (FBS only), week-of-game polls + P4 weighting


from __future__ import annotations
import os
from pathlib import Path
from collections import defaultdict

import pandas as pd
import cfbd
import certifi
from dotenv import load_dotenv

ENV_PATH = Path(__file__).with_name(".env")
load_dotenv(dotenv_path=ENV_PATH) #SHHHHHHH

# ── Config
API_KEY = os.getenv("CFBD_API_KEY")
YEAR = 2025
WEEK = 5  # dont forget to change for next week!!!!!!!!!!!! :D 

# Wins-first knobs
ALPHA_WIN = 0.6          # opponent quality inside wins
ALPHA_LOSS = 0.3         # opponent quality inside losses
BASE_LOSS = 0.90         # base loss penalty magnitude
MARGIN_CAP = 14          # cap for margin effects
MARGIN_WIN_PCT = 0.008   # +0.8% per point (wins)
RELIEF_FLOOR = 0.70      # floor on close-loss relief multiplier
RELIEF_PER_PT = 0.01     # 1% relief per point (losses)
ROAD_MULT, NEUT_MULT, HOME_MULT = 1.04, 1.00, 0.985

# Ranked  multipliers
RANKED_WIN_MAX_BOOST = 0.70      
RANKED_LOSS_MAX_DISCOUNT = 0.40   

# Conference-tier handling for UNRANKED opponents
POWER4_CONFS = {"SEC", "Big Ten", "ACC", "Big 12"}
P4_TEAM_EXCEPTIONS = {"Notre Dame"}  # treat as P4 tier while Independent cause they always good? (JOIN THE ACC!)

# Unranked win scaling by tier
P4_UNRANKED_WIN_SCALE = 0.85  
G5_UNRANKED_WIN_SCALE = 0.40  

# Unranked loss multipliers by tier
P4_UNRANKED_LOSS_MULT = 1.05
G5_UNRANKED_LOSS_MULT = 1.30  

# Season-level SOS multiplier 
def sos_multiplier(avg_opp_q: float) -> float:
    return 0.50 + 1.10 * float(avg_opp_q)

# Diminishing returns for unranked wins (start after the FIRST)
def unranked_wins_decay(extra_unranked_wins: int) -> float:
    extra = max(0, int(extra_unranked_wins))
    return 0.93 ** extra

# Ranked-win additive 
RANKED_WIN_KICKER = 0.30 

# CFBD client (explicit Bearer + TLS CA path for macOS)
configuration = cfbd.Configuration()
configuration.ssl_ca_cert = certifi.where()
api_client = cfbd.ApiClient(configuration)
api_client.set_default_header("Authorization", f"Bearer {API_KEY}")

games_api   = cfbd.GamesApi(api_client)
ratings_api = cfbd.RatingsApi(api_client)
polls_api   = cfbd.RankingsApi(api_client)
teams_api   = cfbd.TeamsApi(api_client)

# ── Helpers
def _norm_team(s: str) -> str:
    return (s or "").strip().lower()

def _fbs_map(year: int) -> dict[str, str]:
    """normalized name -> display name for all FBS schools"""
    teams = teams_api.get_fbs_teams(year=year)
    return {_norm_team(t.school): t.school for t in teams}

def _conference_map(year: int) -> dict[str, str]:
    """display name -> conference (e.g., 'SEC', 'Sun Belt', 'Independent')"""
    teams = teams_api.get_fbs_teams(year=year)
    out = {}
    for t in teams:
        out[t.school] = (t.conference or "").strip()
    return out

def _is_p4_team(team: str, conf_map: dict[str,str]) -> bool:
    conf = conf_map.get(team, "")
    return (conf in POWER4_CONFS) or (team in P4_TEAM_EXCEPTIONS)

def normalize_0_1(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    lo, hi = series.min(), series.max()
    if hi > lo:
        return (series - lo) / (hi - lo)
    return pd.Series(0.5, index=series.index)


# Pull data (FBS only)

def get_games_through_week(year: int, week: int) -> pd.DataFrame:
    """
    Weeks 1..week (FBS only), TWO rows per game (home & away perspectives).
    Columns: week, team, opp, result('W'|'L'|'T'), loc('H'|'A'|'N'), margin(int)
    """
    rows = []
    for w in range(1, week + 1):
        games = games_api.get_games(year=year, week=w, season_type="regular", classification="fbs")
        for g in games:
            if g.home_points is None or g.away_points is None:
                continue
            hp, ap = int(g.home_points), int(g.away_points)
            diff = abs(hp - ap)
            if hp > ap:
                home_res, away_res = "W", "L"
            elif ap > hp:
                home_res, away_res = "L", "W"
            else:
                home_res, away_res = "T", "T"
            rows.append({"week": w, "team": g.home_team, "opp": g.away_team,
                         "result": home_res, "loc": ("N" if g.neutral_site else "H"),
                         "margin": diff})
            rows.append({"week": w, "team": g.away_team, "opp": g.home_team,
                         "result": away_res, "loc": ("N" if g.neutral_site else "A"),
                         "margin": diff})
    return pd.DataFrame(rows, columns=["week","team","opp","result","loc","margin"])

# Polls (week-specific)
def get_coaches_poll_for_week(year: int, week: int) -> pd.DataFrame:
    """
    Coaches poll for EXACT week. Returns: team, rank, points, poll_week
    Name-tolerant: any poll containing 'coach'.
    """
    try:
        rankings = polls_api.get_rankings(year=year, week=week, season_type="regular")
    except Exception:
        return pd.DataFrame(columns=["team","rank","points","poll_week"])
    if not rankings:
        return pd.DataFrame(columns=["team","rank","points","poll_week"])

    entries, found = [], False
    for wk in rankings:
        for poll in getattr(wk, "polls", []):
            name = (poll.poll or "").lower()
            if "coach" in name:
                found = True
                for t in poll.ranks:
                    entries.append({"team": t.school, "rank": t.rank, "points": t.points})
        if found:
            break
    if not entries:
        return pd.DataFrame(columns=["team","rank","points","poll_week"])
    df = pd.DataFrame(entries)
    df["poll_week"] = week
    return df

def build_weekly_coaches_maps(year: int, max_week: int, fbs_keys: set[str]) -> dict[int, tuple[dict[str,float], dict[str,int]]]:
    """
    For each week 1..max_week, build:
      - Q_map_w: display team -> poll quality [0,1]
      - R_map_w: display team -> rank (1..25)
    If a week has no poll, its maps will be empty; we'll fallback later.
    """
    out: dict[int, tuple[dict[str,float], dict[str,int]]] = {}
    for w in range(1, max_week + 1):
        df = get_coaches_poll_for_week(year, w)
        q_map: dict[str,float] = {}
        r_map: dict[str,int] = {}
        if not df.empty:
            df["key"] = df["team"].map(_norm_team)
            df = df[df["key"].isin(fbs_keys)]
            if not df.empty:
                df["Q_rank"] = 1.0 - (df["rank"] - 1) / 24.0
                max_pts = df["points"].max()
                df["Q_pts"]  = (df["points"] / max_pts) if max_pts else 0.0
                df["Q_poll"] = 0.8*df["Q_rank"] + 0.2*df["Q_pts"]
                for _, r in df.iterrows():
                    q_map[r["team"]] = float(r["Q_poll"])
                    r_map[r["team"]] = int(r["rank"])
        out[w] = (q_map, r_map)
    return out

# Fallbacks (SRS + Win% BY WEEK)
def get_srs_rating(year: int) -> pd.DataFrame:
    """Columns: team, srs_rating"""
    srs_list = ratings_api.get_srs(year=year)
    rows = []
    for t in srs_list:
        name = getattr(t, "team", None) or getattr(t, "school", None)
        rating = getattr(t, "rating", None) or getattr(t, "srs", None)
        if name is None or rating is None:
            continue
        rows.append({"team": name, "srs_rating": float(rating)})
    return pd.DataFrame(rows)

def build_win_pct_by_week(games_flat: pd.DataFrame, max_week: int) -> dict[tuple[str,int], float]:
    """
    Return {(team, week) -> win_pct_through_that_week}. Ties=0.5.
    """
    result: dict[tuple[str,int], float] = {}
    for w in range(1, max_week + 1):
        sub = games_flat[games_flat["week"] <= w]
        agg = defaultdict(lambda: {"w":0, "l":0, "t":0})
        for _, r in sub.iterrows():
            if r["result"] == "W":
                agg[r["team"]]["w"] += 1
            elif r["result"] == "L":
                agg[r["team"]]["l"] += 1
            else:
                agg[r["team"]]["t"] += 1
        for team, wl in agg.items():
            gp = wl["w"] + wl["l"] + wl["t"]
            wp = 0.5 if gp == 0 else (wl["w"] + 0.5*wl["t"]) / gp
            result[(team, w)] = wp
    return result

def rank_quality(rank: int | None) -> float:
    if rank is None: return 0.0
    return max(0.0, 1.0 - (rank - 1) / 24.0)

# Per-game value (wins-first) with ranked/unranked + P4/G5 multipliers
def game_value(result: str, opp_Q: float, loc: str, margin: int,
               opp_rank: int | None, opp_is_p4_if_unranked: bool) -> float:
    # location multiplier
    L = ROAD_MULT if loc == "A" else (NEUT_MULT if loc == "N" else HOME_MULT)
    m = min(int(margin), MARGIN_CAP)

    # ranked multipliers
    rq = rank_quality(opp_rank)
    win_rank_boost   = 1.0 + RANKED_WIN_MAX_BOOST * rq
    loss_rank_reduce = 1.0 - RANKED_LOSS_MAX_DISCOUNT * rq

    is_unranked = (opp_rank is None)

    if result == "W":
        M = 1.0 + MARGIN_WIN_PCT * m
        base = (1.0 + ALPHA_WIN * opp_Q) * L * M
        if is_unranked:
            base *= (P4_UNRANKED_WIN_SCALE if opp_is_p4_if_unranked else G5_UNRANKED_WIN_SCALE)
        return base * win_rank_boost

    if result == "L":
        relief = max(RELIEF_FLOOR, 1.0 - RELIEF_PER_PT * m)
        base_penalty = (BASE_LOSS + ALPHA_LOSS * opp_Q) * L * relief
        if is_unranked:
            base_penalty *= (P4_UNRANKED_LOSS_MULT if opp_is_p4_if_unranked else G5_UNRANKED_LOSS_MULT)
        return -base_penalty * loss_rank_reduce

    # tie (dis never happening but just in case!)
    return 0.3 * (1.0 + 0.4 * opp_Q) * L


# Main pipeline

def main() -> None:
    fbs_name_map = _fbs_map(YEAR)         
    conf_map     = _conference_map(YEAR)  
    fbs_keys = set(fbs_name_map.keys())

    # 1) Games 1..W (FBS-only), week-tagged
    games_df = get_games_through_week(YEAR, WEEK)
    if games_df.empty:
        raise RuntimeError("No finished FBS games returned; try another WEEK.")
    games_df["t_key"] = games_df["team"].map(_norm_team)
    games_df["o_key"] = games_df["opp"].map(_norm_team)
    games_df = games_df[games_df["t_key"].isin(fbs_keys) & games_df["o_key"].isin(fbs_keys)].copy()

    # 2) Weekly Coaches maps (Q & rank)
    weekly_poll_maps = build_weekly_coaches_maps(YEAR, WEEK, fbs_keys)  # {w: (Q_map, R_map)}

    # 3) Fallbacks: season SRS (static) and Win% BY WEEK
    srs_df = get_srs_rating(YEAR)
    srs_df["key"] = srs_df["team"].map(_norm_team)
    srs_df = srs_df[srs_df["key"].isin(fbs_keys)]
    srs_Q_map = {}
    if not srs_df.empty:
        srs_df["Q_srs"] = normalize_0_1(srs_df["srs_rating"])
        for _, r in srs_df.iterrows():
            srs_Q_map[r["team"]] = float(r["Q_srs"])

    wp_by_week = build_win_pct_by_week(games_df, WEEK)  # {(team, week): win_pct}

    # 4) Per-game values using WEEK-SPECIFIC rank/quality + P4 tier for UNRANKED
    out_rows = []
    for _, g in games_df.iterrows():
        w = int(g["week"])
        opp_disp  = fbs_name_map.get(g["o_key"], g["opp"])
        team_disp = fbs_name_map.get(g["t_key"], g["team"])

        Q_map_w, R_map_w = weekly_poll_maps.get(w, ({}, {}))

        opp_rank = R_map_w.get(opp_disp)  # None if unranked that week
        if opp_disp in Q_map_w:
            Q = Q_map_w[opp_disp]
        else:
            # Fallback: SRS if available, else shrunk win% through THIS week
            if opp_disp in srs_Q_map:
                Q = srs_Q_map[opp_disp]
            else:
                wp = wp_by_week.get((opp_disp, w), 0.5)
                Q = 0.25 + 0.5 * wp

        opp_is_p4_if_unranked = _is_p4_team(opp_disp, conf_map) if opp_rank is None else False

        val = game_value(g["result"], Q, g["loc"], int(g["margin"]), opp_rank, opp_is_p4_if_unranked)
        out_rows.append({
            "team": team_disp,
            "week": w,
            "game_score": val,
            "opp_Q": Q,
            "ranked_win": (g["result"] == "W" and opp_rank is not None),
            "unranked_win": (g["result"] == "W" and opp_rank is None),
            "ranked_win_Q": (Q if (g["result"] == "W" and opp_rank is not None) else 0.0)
        })

    per_game = pd.DataFrame(out_rows)

    # 5) Aggregate and apply season-level adjustments
    team_scores = (
        per_game.groupby("team", as_index=False)["game_score"].sum()
        .rename(columns={"game_score": "WinsScore"})
    )
    opp_quality = (
        per_game.groupby("team", as_index=False)["opp_Q"].mean()
        .rename(columns={"opp_Q": "AvgOppQ"})
    )
    uw = (
        per_game[per_game["unranked_win"]]
        .groupby("team", as_index=False)["unranked_win"]
        .sum()
        .rename(columns={"unranked_win": "UnrankedWins"})
    )
    rw = (
        per_game[per_game["ranked_win"]]
        .groupby("team", as_index=False)["ranked_win"]
        .sum()
        .rename(columns={"ranked_win": "RankedWins"})
    )
    rq = (
        per_game[per_game["ranked_win"]]
        .groupby("team", as_index=False)["ranked_win_Q"]
        .mean()
        .rename(columns={"ranked_win_Q": "AvgRankQ"})
    )

    combined = (
        team_scores
        .merge(opp_quality, on="team", how="left")
        .merge(uw, on="team", how="left")
        .merge(rw, on="team", how="left")
        .merge(rq, on="team", how="left")
    )
    combined["AvgOppQ"] = combined["AvgOppQ"].fillna(0.5)
    combined["UnrankedWins"] = combined["UnrankedWins"].fillna(0).astype(int)
    combined["RankedWins"] = combined["RankedWins"].fillna(0).astype(int)
    combined["AvgRankQ"] = combined["AvgRankQ"].fillna(0.0)

    # Season-level SOS multiplier (stronger)
    combined["WinsScore"] = combined["WinsScore"] * combined["AvgOppQ"].map(sos_multiplier)

    # Diminishing returns on unranked wins beyond FIRST
    combined["WinsScore"] = combined.apply(
        lambda r: r["WinsScore"] * unranked_wins_decay(r["UnrankedWins"] - 1),
        axis=1
    )

    # Ranked-win additive kicker (scaled by AvgRankQ, 0.7..1.0 factor)
    combined["WinsScore"] = combined["WinsScore"] + RANKED_WIN_KICKER * combined["RankedWins"] * (0.7 + 0.3*combined["AvgRankQ"])

    # Soft clamp: 0 ranked wins + soft schedule → trim a bit
    mask_soft = (combined["RankedWins"] == 0) & (combined["AvgOppQ"] < 0.55)
    combined.loc[mask_soft, "WinsScore"] *= 0.90

    # Sort by WinsScore then schedule toughness
    combined = combined.sort_values(["WinsScore", "AvgOppQ"], ascending=[False, False]).reset_index(drop=True)
    combined.index = combined.index + 1

    top25 = combined[["team", "WinsScore", "AvgOppQ", "RankedWins", "UnrankedWins"]].head(25).copy()

    out_path = Path("cfb_top25.csv")
    top25.to_csv(out_path, index=True)
    print(top25)

    print(f"\n[done] Wrote Top-25 to {out_path.resolve()} (week-of-game polls + P4 weighting)")

if __name__ == "__main__":
    main()
