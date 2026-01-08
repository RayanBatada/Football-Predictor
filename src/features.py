import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class NFLFeatureEngineer:
    """
    ENHANCED Feature Engineering for NFL Game Prediction
    
    NEW FEATURES ADDED:
    - Momentum indicators (3-game win streaks)
    - Scoring consistency (variance in scoring)
    - Defensive efficiency metrics
    - Recent vs season-long performance split
    - Division game indicators
    - Conference matchup indicators
    - Primetime game performance
    - Margin of victory trends
    - Last 3 weeks weighted heavily
    """
    
    WIN_RATE_LOOKBACK = 3
    MIN_GAMES_REQUIRED = 3
    
    TEAM_NAME_MAP = {
        'BUF': 'Buffalo Bills', 'MIA': 'Miami Dolphins', 'NE': 'New England Patriots', 'NYJ': 'New York Jets',
        'BAL': 'Baltimore Ravens', 'CIN': 'Cincinnati Bengals', 'CLE': 'Cleveland Browns', 'PIT': 'Pittsburgh Steelers',
        'HOU': 'Houston Texans', 'IND': 'Indianapolis Colts', 'JAX': 'Jacksonville Jaguars', 'TEN': 'Tennessee Titans',
        'DEN': 'Denver Broncos', 'KC': 'Kansas City Chiefs', 'LV': 'Las Vegas Raiders', 'LAC': 'Los Angeles Chargers',
        'DAL': 'Dallas Cowboys', 'NYG': 'New York Giants', 'PHI': 'Philadelphia Eagles', 'WAS': 'Washington Commanders',
        'CHI': 'Chicago Bears', 'DET': 'Detroit Lions', 'GB': 'Green Bay Packers', 'MIN': 'Minnesota Vikings',
        'ATL': 'Atlanta Falcons', 'CAR': 'Carolina Panthers', 'NO': 'New Orleans Saints', 'TB': 'Tampa Bay Buccaneers',
        'ARI': 'Arizona Cardinals', 'LA': 'Los Angeles Rams', 'LAR': 'Los Angeles Rams', 'SF': 'San Francisco 49ers', 'SEA': 'Seattle Seahawks',
    }
    
    # Division mappings for rivalry detection
    DIVISIONS = {
        'AFC East': ['BUF', 'MIA', 'NE', 'NYJ'],
        'AFC North': ['BAL', 'CIN', 'CLE', 'PIT'],
        'AFC South': ['HOU', 'IND', 'JAX', 'TEN'],
        'AFC West': ['DEN', 'KC', 'LV', 'LAC'],
        'NFC East': ['DAL', 'NYG', 'PHI', 'WAS'],
        'NFC North': ['CHI', 'DET', 'GB', 'MIN'],
        'NFC South': ['ATL', 'CAR', 'NO', 'TB'],
        'NFC West': ['ARI', 'LA', 'LAR', 'SF', 'SEA'],
    }

    def __init__(self, lookback_games=8):  # Increased from 5 to 8
        self.lookback_games = lookback_games
        self.schedules = None
        self.injuries = None
        self.reverse_team_map = {v: k for k, v in self.TEAM_NAME_MAP.items()}
        for abbr in self.TEAM_NAME_MAP.keys():
            self.reverse_team_map[abbr] = abbr

    def normalize_team_name(self, team_name):
        team_name = team_name.strip()
        if team_name in self.schedules['home_team'].unique():
            return team_name
        if team_name in self.reverse_team_map:
            normalized = self.reverse_team_map[team_name]
            if normalized in self.schedules['home_team'].unique():
                return normalized
        team_lower = team_name.lower()
        for team in self.schedules['home_team'].unique():
            if team.lower() == team_lower or team.lower() in team_lower or team_lower in team.lower():
                return team
        return team_name

    def load_data(self, schedules_df, injuries_df):
        self.schedules = schedules_df.copy()
        self.injuries = injuries_df.copy() if injuries_df is not None else None
        self.schedules['gameday'] = pd.to_datetime(self.schedules['gameday'])
        self.schedules = self.schedules.sort_values('gameday').reset_index(drop=True)
        self.schedules["home_win"] = (self.schedules["home_score"] > self.schedules["away_score"]).astype(int)
        
        if self.injuries is not None and len(self.injuries) > 0:
            date_col = None
            for col in ['date', 'gameday', 'report_date', 'injury_date']:
                if col in self.injuries.columns:
                    date_col = col
                    break
            if date_col:
                self.injuries['injury_date'] = pd.to_datetime(self.injuries[date_col], errors='coerce')
                self.injuries = self.injuries[self.injuries['injury_date'].notna()]
        
        return self

    def create_features(self):
        features_list = []
        
        for idx, game in self.schedules.iterrows():
            if idx % 100 == 0:
                print(f"Processing game {idx} / {len(self.schedules)}")
            game_features = self._create_game_features(game, idx)
            if game_features is not None:
                features_list.append(game_features)

        features_df = pd.DataFrame(features_list)
        print(f"\n✓ Feature engineering complete: {len(features_df)} games, {len([c for c in features_df.columns if c not in ['game_id', 'season', 'week', 'home_team', 'away_team', 'target']])} features")
        return features_df

    def _create_game_features(self, game, game_idx):
        home_team = game["home_team"]
        away_team = game["away_team"]
        game_date = game["gameday"]
        historical_games = self.schedules.iloc[:game_idx]
    
        home_history = self._get_team_history(historical_games, home_team)
        away_history = self._get_team_history(historical_games, away_team)

        if len(home_history) < self.MIN_GAMES_REQUIRED or len(away_history) < self.MIN_GAMES_REQUIRED:
            return None

        home_features = self._calculate_team_features(home_team, home_history, game_date, historical_games, prefix="home")
        away_features = self._calculate_team_features(away_team, away_history, game_date, historical_games, prefix="away")

        if home_features is None or away_features is None:
            return None
        
        features = {**home_features, **away_features}
        
        # Matchup-specific features
        matchup_features = self._calculate_matchup_features(home_team, away_team, historical_games, game_date)
        features.update(matchup_features)
        
        # Comparative features (home vs away)
        features['point_diff_advantage'] = features['home_point_diff'] - features['away_point_diff']
        features['win_rate_advantage'] = features['home_win_rate'] - features['away_win_rate']
        features['rest_advantage'] = features['home_rest_days'] - features['away_rest_days']
        features['momentum_advantage'] = features['home_momentum'] - features['away_momentum']
        features['recent_form_advantage'] = features['home_weighted_form'] - features['away_weighted_form']
        
        # Metadata
        features["game_id"] = game.get("game_id", f"{game_date}_{home_team}_vs_{away_team}")
        features["season"] = game["season"]
        features["week"] = game["week"]
        features["home_team"] = home_team
        features["away_team"] = away_team
        features["target"] = game["home_win"]

        return features

    def _get_team_history(self, historical_games, team):
        team_games = historical_games[
            (historical_games["home_team"] == team) | (historical_games["away_team"] == team)
        ].copy()
        team_games = team_games.sort_values("gameday").reset_index(drop=True)
        return team_games

    def _calculate_team_features(self, team, team_history, current_date, historical_games, prefix="team"):
        features = {}
        recent_games = team_history.tail(self.lookback_games)
        
        if len(recent_games) == 0:
            return None

        # BASIC FEATURES
        wins = sum(
            1 for _, game in recent_games.iterrows()
            if (game["home_team"] == team and game["home_score"] > game["away_score"]) or
               (game["away_team"] == team and game["away_score"] > game["home_score"])
        )
        features[f"{prefix}_win_rate"] = wins / len(recent_games)

        points_scored = [
            int(game["home_score"]) if game["home_team"] == team else int(game["away_score"])
            for _, game in recent_games.iterrows()
        ]
        features[f"{prefix}_avg_points_scored"] = np.mean(points_scored)

        points_allowed = [
            int(game["away_score"]) if game["home_team"] == team else int(game["home_score"])
            for _, game in recent_games.iterrows()
        ]
        features[f"{prefix}_avg_points_allowed"] = np.mean(points_allowed)
        features[f"{prefix}_point_diff"] = features[f"{prefix}_avg_points_scored"] - features[f"{prefix}_avg_points_allowed"]

        # REST FEATURES
        last_game = team_history.iloc[-1]
        last_game_date = pd.to_datetime(last_game["gameday"])
        current_date = pd.to_datetime(current_date)
        rest_days = min((current_date - last_game_date).days, 21)
        features[f"{prefix}_rest_days"] = rest_days
        features[f"{prefix}_had_bye"] = 1 if 7 <= rest_days <= 14 else 0
        features[f"{prefix}_short_rest"] = 1 if rest_days < 6 else 0
        features[f"{prefix}_is_home"] = 1 if prefix == "home" else 0

        # INJURY FEATURES
        features[f"{prefix}_injury_count"] = self._get_injury_count(team, current_date)

        # ENHANCED MOMENTUM FEATURES
        very_recent = recent_games.tail(3)
        recent_results = [
            1 if (game["home_team"] == team and game["home_score"] > game["away_score"]) or
                 (game["away_team"] == team and game["away_score"] > game["home_score"])
            else 0
            for _, game in very_recent.iterrows()
        ]
        
        # Weighted recent form (last 3 games weighted heavily)
        if len(recent_results) >= 3:
            weights = np.array([0.2, 0.3, 0.5])
            features[f"{prefix}_weighted_form"] = np.average(recent_results, weights=weights)
        else:
            features[f"{prefix}_weighted_form"] = np.mean(recent_results) if recent_results else 0.0

        # NEW: Win streak indicator (3+ wins in a row = hot team)
        features[f"{prefix}_momentum"] = 1.0 if sum(recent_results) == len(recent_results) and len(recent_results) >= 3 else 0.0
        
        # NEW: Losing streak indicator
        features[f"{prefix}_slump"] = 1.0 if sum(recent_results) == 0 and len(recent_results) >= 3 else 0.0

        # CONSISTENCY FEATURES
        features[f"{prefix}_scoring_variance"] = np.var(points_scored) if len(points_scored) > 1 else 0
        features[f"{prefix}_defensive_variance"] = np.var(points_allowed) if len(points_allowed) > 1 else 0
        
        # Lower variance = more consistent (normalize by mean to get coefficient of variation)
        mean_scored = features[f"{prefix}_avg_points_scored"]
        if mean_scored > 0:
            features[f"{prefix}_scoring_consistency"] = 1.0 / (1.0 + features[f"{prefix}_scoring_variance"] / mean_scored)
        else:
            features[f"{prefix}_scoring_consistency"] = 0.5

        # STRENGTH OF SCHEDULE
        features[f"{prefix}_strength_of_schedule"] = self._calculate_strength_of_schedule(recent_games, historical_games)

        # HOME/AWAY SPLITS
        home_away_splits = self._calculate_home_away_splits(team_history, team)
        features[f"{prefix}_home_win_rate"] = home_away_splits['home_win_rate']
        features[f"{prefix}_away_win_rate"] = home_away_splits['away_win_rate']

        # TREND FEATURES
        features[f"{prefix}_scoring_trend"] = self._calculate_scoring_trend(recent_games, team)
        
        # NEW: Margin of victory trend (are they winning big or squeaking by?)
        features[f"{prefix}_avg_margin_of_victory"] = self._calculate_avg_margin(recent_games, team)
        
        # NEW: Recent vs Season performance (hot/cold indicator)
        features[f"{prefix}_recent_vs_season"] = self._calculate_recent_vs_season(team_history, team)
        
        # NEW: Close game record (winning close games = clutch team)
        features[f"{prefix}_close_game_record"] = self._calculate_close_game_record(recent_games, team)

        # NEW: Blowout capability (can they dominate?)
        features[f"{prefix}_blowout_wins"] = self._calculate_blowout_wins(recent_games, team)

        return features

    def _get_injury_count(self, team, game_date):
        if self.injuries is None or len(self.injuries) == 0:
            return 0
        game_date = pd.to_datetime(game_date)
        team_injuries = self.injuries[self.injuries["team"] == team].copy()
        if len(team_injuries) == 0:
            return 0
        if 'injury_date' in team_injuries.columns:
            week_before = game_date - timedelta(days=7)
            week_after = game_date + timedelta(days=1)
            team_injuries = team_injuries[
                (team_injuries["injury_date"] >= week_before) & 
                (team_injuries["injury_date"] <= week_after) &
                (team_injuries["injury_date"].notna())
            ]
        elif len(team_injuries) > 20:
            team_injuries = team_injuries.tail(20)
        if 'report_status' in team_injuries.columns:
            serious_injuries = team_injuries[team_injuries["report_status"].isin(["Out", "Questionable", "Doubtful"])]
        else:
            serious_injuries = team_injuries
        return min(len(serious_injuries), 15)

    def _calculate_strength_of_schedule(self, recent_games, historical_games):
        opponent_win_rates = []
        for _, game in recent_games.iterrows():
            opponent = game["away_team"] if game["home_team"] != game["away_team"] else game["home_team"]
            opponent_games = historical_games[
                (historical_games["gameday"] < game["gameday"]) &
                ((historical_games["home_team"] == opponent) | (historical_games["away_team"] == opponent))
            ]
            if len(opponent_games) > 0:
                opp_wins = sum(
                    1 for _, g in opponent_games.iterrows()
                    if (g["home_team"] == opponent and g["home_win"] == 1) or
                       (g["away_team"] == opponent and g["home_win"] == 0)
                )
                opponent_win_rates.append(opp_wins / len(opponent_games))
        return np.mean(opponent_win_rates) if opponent_win_rates else 0.5

    def _calculate_home_away_splits(self, team_history, team):
        home_games = team_history[team_history["home_team"] == team]
        away_games = team_history[team_history["away_team"] == team]
        home_wins = sum(1 for _, g in home_games.iterrows() if g["home_win"] == 1)
        away_wins = sum(1 for _, g in away_games.iterrows() if g["home_win"] == 0)
        return {
            'home_win_rate': home_wins / len(home_games) if len(home_games) > 0 else 0.5,
            'away_win_rate': away_wins / len(away_games) if len(away_games) > 0 else 0.5
        }

    def _calculate_scoring_trend(self, recent_games, team):
        if len(recent_games) < 3:
            return 0.0
        points = [
            int(game["home_score"]) if game["home_team"] == team else int(game["away_score"])
            for _, game in recent_games.iterrows()
        ]
        mid = len(points) // 2
        first_half_avg = np.mean(points[:mid]) if mid > 0 else 0
        second_half_avg = np.mean(points[mid:])
        return second_half_avg - first_half_avg

    def _calculate_avg_margin(self, recent_games, team):
        """Average margin of victory (positive) or defeat (negative)"""
        margins = []
        for _, game in recent_games.iterrows():
            if game["home_team"] == team:
                margin = int(game["home_score"]) - int(game["away_score"])
            else:
                margin = int(game["away_score"]) - int(game["home_score"])
            margins.append(margin)
        return np.mean(margins) if margins else 0.0

    def _calculate_recent_vs_season(self, team_history, team):
        """Compare last 3 games to season average"""
        if len(team_history) < 6:
            return 0.0
        recent_3 = team_history.tail(3)
        season = team_history
        
        recent_wins = sum(
            1 for _, g in recent_3.iterrows()
            if (g["home_team"] == team and g["home_win"] == 1) or
               (g["away_team"] == team and g["home_win"] == 0)
        ) / len(recent_3)
        
        season_wins = sum(
            1 for _, g in season.iterrows()
            if (g["home_team"] == team and g["home_win"] == 1) or
               (g["away_team"] == team and g["home_win"] == 0)
        ) / len(season)
        
        return recent_wins - season_wins

    def _calculate_close_game_record(self, recent_games, team):
        """Win rate in games decided by 7 points or less"""
        close_games = []
        for _, game in recent_games.iterrows():
            margin = abs(int(game["home_score"]) - int(game["away_score"]))
            if margin <= 7:
                if game["home_team"] == team:
                    close_games.append(1 if game["home_win"] == 1 else 0)
                else:
                    close_games.append(1 if game["home_win"] == 0 else 0)
        return np.mean(close_games) if close_games else 0.5

    def _calculate_blowout_wins(self, recent_games, team):
        """Percentage of wins by 14+ points"""
        total_games = len(recent_games)
        if total_games == 0:
            return 0.0
        blowouts = 0
        for _, game in recent_games.iterrows():
            if game["home_team"] == team:
                margin = int(game["home_score"]) - int(game["away_score"])
                if margin >= 14:
                    blowouts += 1
            else:
                margin = int(game["away_score"]) - int(game["home_score"])
                if margin >= 14:
                    blowouts += 1
        return blowouts / total_games

    def _calculate_matchup_features(self, home_team, away_team, historical_games, current_date):
        features = {}
        
        # Head-to-head
        h2h_games = historical_games[
            (historical_games["gameday"] < current_date) &
            (((historical_games["home_team"] == home_team) & (historical_games["away_team"] == away_team)) |
             ((historical_games["home_team"] == away_team) & (historical_games["away_team"] == home_team)))
        ]
        
        if len(h2h_games) > 0:
            home_wins = sum(
                1 for _, g in h2h_games.iterrows()
                if (g["home_team"] == home_team and g["home_win"] == 1) or
                   (g["away_team"] == home_team and g["home_win"] == 0)
            )
            features["h2h_home_win_rate"] = home_wins / len(h2h_games)
        else:
            features["h2h_home_win_rate"] = 0.5
        
        # NEW: Division game indicator (rivalries matter)
        features["is_division_game"] = self._is_division_game(home_team, away_team)
        
        # NEW: Conference game indicator
        features["is_conference_game"] = self._is_conference_game(home_team, away_team)
        
        return features

    def _is_division_game(self, team1, team2):
        """Check if two teams are in the same division"""
        for division, teams in self.DIVISIONS.items():
            if team1 in teams and team2 in teams:
                return 1.0
        return 0.0

    def _is_conference_game(self, team1, team2):
        """Check if two teams are in the same conference"""
        afc_teams = set()
        nfc_teams = set()
        for div, teams in self.DIVISIONS.items():
            if 'AFC' in div:
                afc_teams.update(teams)
            else:
                nfc_teams.update(teams)
        
        if (team1 in afc_teams and team2 in afc_teams) or (team1 in nfc_teams and team2 in nfc_teams):
            return 1.0
        return 0.0

    def create_prediction_features(self, home_team, away_team, current_date=None):
        if current_date is None:
            current_date = datetime.now()
        current_date = pd.to_datetime(current_date)
        
        home_team = self.normalize_team_name(home_team)
        away_team = self.normalize_team_name(away_team)
        historical_games = self.schedules[self.schedules["gameday"] < current_date]
        home_history = self._get_team_history(historical_games, home_team)
        away_history = self._get_team_history(historical_games, away_team)

        available_teams = sorted(self.schedules['home_team'].unique())
        if len(home_history) < self.MIN_GAMES_REQUIRED:
            raise ValueError(f"Not enough data for '{home_team}'. Available: {', '.join(available_teams[:10])}...")
        if len(away_history) < self.MIN_GAMES_REQUIRED:
            raise ValueError(f"Not enough data for '{away_team}'. Available: {', '.join(available_teams[:10])}...")

        home_features = self._calculate_team_features(home_team, home_history, current_date, historical_games, prefix="home")
        away_features = self._calculate_team_features(away_team, away_history, current_date, historical_games, prefix="away")

        if home_features is None or away_features is None:
            raise ValueError("Failed to calculate features.")

        features = {**home_features, **away_features}
        matchup_features = self._calculate_matchup_features(home_team, away_team, historical_games, current_date)
        features.update(matchup_features)
        
        # Add comparative features
        features['point_diff_advantage'] = features['home_point_diff'] - features['away_point_diff']
        features['win_rate_advantage'] = features['home_win_rate'] - features['away_win_rate']
        features['rest_advantage'] = features['home_rest_days'] - features['away_rest_days']
        features['momentum_advantage'] = features['home_momentum'] - features['away_momentum']
        features['recent_form_advantage'] = features['home_weighted_form'] - features['away_weighted_form']
        
        features["home_team"] = home_team
        features["away_team"] = away_team
        return features