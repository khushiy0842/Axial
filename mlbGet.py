#!/usr/bin/env python3
"""
Build an MLB rest-days dataset for one season using pybaseball.

Rest day (inferred) = team played, player did NOT appear that day.
Optionally merge an IL file later to exclude injury days.

Outputs:
  data/mlb_<YEAR>_rest_days.csv

Requires:
  pip install pybaseball pandas numpy tqdm
"""

import os
import sys
import time
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from pybaseball import (
    schedule_and_record,   # team schedule by year
    team_batting,          # team-season batting totals (to pick top hitters)
    playerid_lookup,       # map name -> bbref id
    season_game_logs       # per-player game logs by season
)

# ----------------------------
# Config
# ----------------------------
YEAR = 2023                        # <--- change as needed
TOP_N_BATTERS_PER_TEAM = 12        # keep runtime reasonable
OUTDIR = "data"
OUTFILE = f"{OUTDIR}/mlb_{YEAR}_rest_days.csv"
INJURY_CSV = None                  # e.g., "data/injuries_2023.csv" with cols: player_id_bbref,date,on_IL

PAUSE_S = 1.0       # polite delay between requests

# Stable set of modern MLB team codes that pybaseball accepts for recent seasons
MLB_TEAMS = [
    "ARI","ATL","BAL","BOS","CHC","CHW","CIN","CLE","COL",
    "DET","HOU","KCR","LAA","LAD","MIA","MIL","MIN","NYM",
    "NYY","OAK","PHI","PIT","SDP","SEA","SFG","STL","TBR",
    "TEX","TOR","WSN"
]
# Note: older seasons might use "FLA"/"TBD"/"MON" etc. For 2023+ these are fine.


def get_team_schedule(team_code: str, year: int) -> pd.DataFrame:
    """Return team schedule with standardized columns."""
    try:
        df = schedule_and_record(team_code, year)
        # Expected columns include 'Date' and 'Tm'
        df = df.rename(columns={"Tm": "team"})
        # Normalize date (strip asterisks if present)
        df["Date"] = df["Date"].astype(str).str.replace("*", "", regex=False)
        df["game_date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["game_date"])
        return df[["game_date", "team"]].assign(team_code=team_code, team_played=1)
    except Exception as e:
        print(f"[warn] schedule failed for {team_code} {year}: {e}")
        return pd.DataFrame(columns=["game_date","team","team_code","team_played"])


def get_top_batters(team_code: str, year: int, top_n: int) -> pd.DataFrame:
    """Pick top-N hitters by PA from team-season totals."""
    try:
        tb = team_batting(year, team_code)
        # Expect columns: 'Name', 'PA'
        tb = tb.rename(columns={"Name": "player_name"})
        tb = tb.loc[tb["PA"].fillna(0) > 0].copy()
        tb["player_name"] = tb["player_name"].str.strip()
        tb = tb.sort_values("PA", ascending=False).head(top_n).reset_index(drop=True)
        tb["team_code"] = team_code
        return tb[["player_name","PA","team_code"]]
    except Exception as e:
        print(f"[warn] team_batting failed for {team_code} {year}: {e}")
        return pd.DataFrame(columns=["player_name","PA","team_code"])


def name_to_bbref_id(player_name: str) -> str:
    """
    Map 'First Last' to bbref slug (e.g., 'troutmi01') using playerid_lookup.
    If multiple rows, prefer most recent MLB service.
    """
    try:
        parts = player_name.split()
        if len(parts) < 2:
            return ""
        last = parts[-1]
        first = " ".join(parts[:-1])
        df = playerid_lookup(last, first)
        if df.empty:
            return ""
        # Prefer those with MLB service and latest 'mlb_played_last'
        if "mlb_played_last" in df.columns:
            df = df.sort_values(["mlb_played_last","mlb_played_first"], ascending=[False, False])
        slug = df["key_bbref"].dropna().astype(str).head(1).values
        return slug[0] if len(slug) else ""
    except Exception:
        return ""


def fetch_player_game_logs_bbref(bbref_id: str, year: int) -> pd.DataFrame:
    """
    Pull per-game logs for a player-season via season_game_logs (pybaseball).
    Returns minimal columns needed.
    """
    try:
        g = season_game_logs(bbref_id, year)
        # Normalize columns
        g = g.rename(columns={
            "Date": "game_date",
            "Tm": "team",
            "Opp": "opp",
        })
        g["game_date"] = pd.to_datetime(g["game_date"], errors="coerce")
        g = g.dropna(subset=["game_date"])
        g["appeared"] = 1
        g["player_id_bbref"] = bbref_id

        keep = ["player_id_bbref","game_date","team","opp","appeared"]
        if "PA" in g.columns:
            keep.append("PA")
        g = g[keep].drop_duplicates()
        if "PA" not in g.columns:
            g["PA"] = np.nan
        return g
    except Exception as e:
        print(f"[warn] logs failed for {bbref_id} {year}: {e}")
        return pd.DataFrame(columns=["player_id_bbref","game_date","team","opp","appeared","PA"])


def attach_injuries(panel: pd.DataFrame, injury_csv: str) -> pd.DataFrame:
    """
    Optionally merge an IL CSV with columns: player_id_bbref,date,on_IL
    """
    inj = pd.read_csv(injury_csv)
    inj = inj.rename(columns={"date": "game_date"})
    inj["game_date"] = pd.to_datetime(inj["game_date"], errors="coerce")
    inj = inj.dropna(subset=["game_date"])
    return panel.merge(inj[["player_id_bbref","game_date","on_IL"]],
                       on=["player_id_bbref","game_date"], how="left")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"[+] Building MLB rest dataset for {YEAR}")

    # 1) Schedules
    print("[1/5] Pulling team schedules…")
    sched_list: List[pd.DataFrame] = []
    for code in MLB_TEAMS:
        s = get_team_schedule(code, YEAR)
        if not s.empty:
            sched_list.append(s)
        time.sleep(PAUSE_S)
    if not sched_list:
        print("No schedules found; abort.")
        sys.exit(1)
    sched = pd.concat(sched_list, ignore_index=True).drop_duplicates()

    # 2) Top batters per team
    print("[2/5] Selecting top batters by team…")
    roster_list: List[pd.DataFrame] = []
    for code in tqdm(MLB_TEAMS):
        tb = get_top_batters(code, YEAR, TOP_N_BATTERS_PER_TEAM)
        if not tb.empty:
            roster_list.append(tb)
        time.sleep(PAUSE_S)
    if not roster_list:
        print("No team batting data; abort.")
        sys.exit(1)
    roster = (pd.concat(roster_list, ignore_index=True)
                .drop_duplicates(subset=["player_name","team_code"]))

    # 3) Resolve names -> bbref ids
    print("[3/5] Resolving player ids…")
    roster["player_id_bbref"] = roster["player_name"].apply(name_to_bbref_id)
    roster = roster[roster["player_id_bbref"].astype(bool)]
    print(f"   Resolved {roster['player_id_bbref'].nunique()} players.")

    # 4) Player game logs
    print("[4/5] Fetching player game logs (this can take a while)…")
    logs_list: List[pd.DataFrame] = []
    for bbref_id in tqdm(roster["player_id_bbref"].unique()):
        g = fetch_player_game_logs_bbref(bbref_id, YEAR)
        if not g.empty:
            logs_list.append(g)
        time.sleep(PAUSE_S)
    if not logs_list:
        print("No player logs fetched; abort.")
        sys.exit(1)
    games = pd.concat(logs_list, ignore_index=True)

    # 5) Infer rest: for each player's teams, mark dates the team played but player didn't appear
    print("[5/5] Inferring rest days…")
    # Which teams did the player appear for (covers mid-season team changes)
    player_teams = (games.groupby(["player_id_bbref","team"])["game_date"]
                         .min().reset_index()[["player_id_bbref","team"]])

    # player x (their teams' schedules)
    player_dates = player_teams.merge(
        sched[["game_date","team","team_played"]],
        on="team", how="left"
    ).dropna(subset=["game_date"])

    # appearances
    appearances = games[["player_id_bbref","game_date","appeared"]].drop_duplicates()
    panel = player_dates.merge(appearances, on=["player_id_bbref","game_date"], how="left")
    panel["appeared"] = panel["appeared"].fillna(0).astype(int)
    panel["rest_flag"] = ((panel["team_played"] == 1) & (panel["appeared"] == 0)).astype(int)

    # (Optional) merge IL days
    if INJURY_CSV and os.path.exists(INJURY_CSV):
        print("[-] Merging IL file…")
        panel = attach_injuries(panel, INJURY_CSV)
        panel["on_IL"] = panel["on_IL"].fillna(0).astype(int)
        panel.loc[panel["on_IL"] == 1, "rest_flag"] = 0
    else:
        panel["on_IL"] = 0

    # fatigue covariates
    panel = panel.sort_values(["player_id_bbref","game_date"]).reset_index(drop=True)
    panel["days_since_last_game"] = panel.groupby("player_id_bbref")["game_date"].diff().dt.days
    panel["prev_day_was_rest"] = panel.groupby("player_id_bbref")["rest_flag"].shift(1).fillna(0).astype(int)

    # add per-game PA (if present in logs)
    perf = games[["player_id_bbref","game_date","PA"]].drop_duplicates() if "PA" in games.columns else None
    if perf is not None:
        panel = panel.merge(perf, on=["player_id_bbref","game_date"], how="left")
        panel["PA"] = panel["PA"].fillna(0)
    else:
        panel["PA"] = 0

    os.makedirs(OUTDIR, exist_ok=True)
    panel.to_csv(OUTFILE, index=False)
    print(f"[+] Wrote {OUTFILE}  ({len(panel):,} rows)")
    print("Columns:", list(panel.columns))
    print("\nTip: next, merge Statcast per-game metrics (xwOBA, EV, LA) for a stronger NN target.")
    print("     If you need IL dates auto-fetched, I can add a quick scraper for ProSportsTransactions.")
    

if __name__ == "__main__":
    main()
