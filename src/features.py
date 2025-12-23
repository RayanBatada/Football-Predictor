import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class NFLFeatureEngineer:
    """
    Engineer features for NFL game prediction with advanced feature engineering
    
    CRITICAL FIXES:
    - Fixed injury date filtering (was completely broken)
    - Added team name normalization (handles abbreviations and full names)
    - Added data staleness warnings
    - Added data quality validation
    - Better error messages
    """
    
    WIN_RATE_LOOKBACK = 3
    MIN_GAMES_REQUIRED = 3
    
    # NFL Team Name Mapping (abbreviation -> full name)
    TEAM_NAME_MAP = {
        # AFC East
        'BUF': 'Buffalo Bills',
        'MIA': 'Miami Dolphins', 
        'NE': 'New England Patriots',
        'NYJ': 'New York Jets',
        
        # AFC North
        'BAL': 'Baltimore Ravens',
        'CIN': 'Cincinnati Bengals',
        'CLE': 'Cleveland Browns',
        'PIT': 'Pittsburgh Steelers',
        
        # AFC South
        'HOU': 'Houston Texans',
        'IND': 'Indianapolis Colts',
        'JAX': 'Jacksonville Jaguars',
        'TEN': 'Tennessee Titans',
        
        # AFC West
        'DEN': 'Denver Broncos',
        'KC': 'Kansas City Chiefs',
        'LV': 'Las Vegas Raiders',
        'LAC': 'Los Angeles Chargers',
        
        # NFC East
        'DAL': 'Dallas Cowboys',
        'NYG': 'New York Giants',
        'PHI': 'Philadelphia Eagles',
        'WAS': 'Washington Commanders',
        
        # NFC North
        'CHI': 'Chicago Bears',
        'DET': 'Detroit Lions',
        'GB': 'Green Bay Packers',
        'MIN': 'Minnesota Vikings',
        
        # NFC South
        'ATL': 'Atlanta Falcons',
        'CAR': 'Carolina Panthers',
        'NO': 'New Orleans Saints',
        'TB': 'Tampa Bay Buccaneers',
        
        # NFC West
        'ARI': 'Arizona Cardinals',
        'LA': 'Los Angeles Rams',
        'LAR': 'Los Angeles Rams',
        'SF': 'San Francisco 49ers',
        'SEA': 'Seattle Seahawks',
    }

    def __init__(self, lookback_games=8):
        self.lookback_games = lookback_games
        self.schedules = None
        self.injuries = None
        
        # Create reverse mapping (full name -> abbreviation)
        self.reverse_team_map = {v: k for k, v in self.TEAM_NAME_MAP.items()}
        # Add abbreviations mapping to themselves
        for abbr in self.TEAM_NAME_MAP.keys():
            self.reverse_team_map[abbr] = abbr

    def normalize_team_name(self, team_name):
        """
        Convert any team name format to the format used in your data
        """
        team_name = team_name.strip()
        
        # Check if it's already in the schedules data
        if team_name in self.schedules['home_team'].unique():
            return team_name
        
        # Try exact match in reverse map
        if team_name in self.reverse_team_map:
            normalized = self.reverse_team_map[team_name]
            if normalized in self.schedules['home_team'].unique():
                return normalized
        
        # Try case-insensitive partial match
        team_lower = team_name.lower()
        for team in self.schedules['home_team'].unique():
            if team.lower() == team_lower or team.lower() in team_lower or team_lower in team.lower():
                return team
        
        # If nothing works, return original and let it fail with good error message
        return team_name

    def load_data(self, schedules_df, injuries_df):
        """Load and prepare data with proper date handling"""
        self.schedules = schedules_df.copy()
        self.injuries = injuries_df.copy() if injuries_df is not None else None

        # Sort by date to ensure chronological order
        self.schedules['gameday'] = pd.to_datetime(self.schedules['gameday'])
        self.schedules = self.schedules.sort_values('gameday').reset_index(drop=True)
        
        # Create home win indicator
        self.schedules["home_win"] = (self.schedules["home_score"] > self.schedules["away_score"]).astype(int)
        
        # CRITICAL FIX: Prepare injuries data properly
        if self.injuries is not None and len(self.injuries) > 0:
            # Try multiple possible date column names
            date_col = None
            for col in ['date', 'gameday', 'report_date', 'injury_date']:
                if col in self.injuries.columns:
                    date_col = col
                    break
            
            if date_col:
                self.injuries['injury_date'] = pd.to_datetime(self.injuries[date_col], errors='coerce')
                # Remove rows with invalid dates
                self.injuries = self.injuries[self.injuries['injury_date'].notna()]
                print(f"  - Processed {len(self.injuries)} injury records with valid dates")
            else:
                print("  ⚠ Warning: No date column found in injuries data. Injury feature will be unreliable.")
                self.injuries['injury_date'] = pd.NaT
        
        # Data quality check
        self._validate_data_quality()

        return self

    def _validate_data_quality(self):
        """Check for data quality issues"""
        print("\n  Data Quality Check:")
        
        # Check date range
        min_date = self.schedules['gameday'].min()
        max_date = self.schedules['gameday'].max()
        days_span = (max_date - min_date).days
        
        print(f"    Date range: {min_date.date()} to {max_date.date()} ({days_span} days)")
        
        # Warn if data is old
        days_since_last_game = (datetime.now() - max_date).days
        if days_since_last_game > 30:
            print(f"    ⚠ WARNING: Most recent game is {days_since_last_game} days old!")
            print(f"    ⚠ Predictions will be based on outdated data")
        
        # Check for missing scores
        missing = self.schedules[['home_score', 'away_score']].isna().sum().sum()
        if missing > 0:
            print(f"    ⚠ Warning: {missing} missing scores")
        
        # Check injury data
        if self.injuries is not None:
            if 'injury_date' in self.injuries.columns:
                valid_dates = self.injuries['injury_date'].notna().sum()
                print(f"    Injury records with valid dates: {valid_dates}/{len(self.injuries)}")
            else:
                print(f"    ⚠ Warning: Injury data has no date information")

    def create_features(self):
        """Create features for all games in historical data"""
        features_list = []
        
        for idx, game in self.schedules.iterrows():
            if idx % 100 == 0:
                print(f"Processing game {idx} / {len(self.schedules)}")

            game_features = self._create_game_features(game, idx)

            if game_features is not None:
                features_list.append(game_features)

        features_df = pd.DataFrame(features_list)

        print("\nFeature engineering complete.")
        print(f"Created features for {len(features_df)} games.")
        print(f"Number of features: {len([col for col in features_df.columns if col not in ['game_id', 'season', 'week', 'home_team', 'away_team', 'target']])}")

        return features_df

    def _create_game_features(self, game, game_idx):
        """Calculate all features for a single game"""
        home_team = game["home_team"]
        away_team = game["away_team"]
        game_date = game["gameday"]

        # Get only games that happened before this one
        historical_games = self.schedules.iloc[:game_idx]
    
        home_history = self._get_team_history(historical_games, home_team)
        away_history = self._get_team_history(historical_games, away_team)

        # Need minimum games for both teams
        if len(home_history) < self.MIN_GAMES_REQUIRED or len(away_history) < self.MIN_GAMES_REQUIRED:
            return None

        # Calculate features for both teams
        home_features = self._calculate_team_features(
            home_team, home_history, game_date, historical_games, prefix="home"
        )
        away_features = self._calculate_team_features(
            away_team, away_history, game_date, historical_games, prefix="away"
        )

        if home_features is None or away_features is None:
            return None
        
        # Combine features
        features = {**home_features, **away_features}
        
        # Add matchup-specific features
        matchup_features = self._calculate_matchup_features(
            home_team, away_team, historical_games, game_date
        )
        features.update(matchup_features)
        
        # Add metadata
        features["game_id"] = game.get("game_id", f"{game_date}_{home_team}_vs_{away_team}")
        features["season"] = game["season"]
        features["week"] = game["week"]
        features["home_team"] = home_team
        features["away_team"] = away_team
        features["target"] = game["home_win"]

        return features

    def _get_team_history(self, historical_games, team):
        """Get all games a team played, sorted by date"""
        team_games = historical_games[
            (historical_games["home_team"] == team) | (historical_games["away_team"] == team)
        ].copy()

        team_games = team_games.sort_values("gameday").reset_index(drop=True)
        return team_games

    def _get_injury_count(self, team, game_date):
        """
        CRITICAL FIX: Now properly filters injuries by date AND team
        Count serious injuries (Out/Questionable) within a week of game date
        """
        if self.injuries is None or len(self.injuries) == 0:
            return 0

        game_date = pd.to_datetime(game_date)

        # CRITICAL: Filter by team FIRST
        team_injuries = self.injuries[self.injuries["team"] == team].copy()
        
        if len(team_injuries) == 0:
            return 0

        # CRITICAL: Filter by date if injury_date column exists
        if 'injury_date' in team_injuries.columns:
            # Only count injuries within 7 days before game
            week_before = game_date - timedelta(days=7)
            week_after = game_date + timedelta(days=1)  # Only 1 day after, not 7
            
            team_injuries = team_injuries[
                (team_injuries["injury_date"] >= week_before) & 
                (team_injuries["injury_date"] <= week_after) &
                (team_injuries["injury_date"].notna())
            ]
        
        # If no date filtering possible, only count most recent injuries (last 20)
        elif len(team_injuries) > 20:
            team_injuries = team_injuries.tail(20)

        # Count serious injuries only
        if 'report_status' in team_injuries.columns:
            serious_injuries = team_injuries[
                team_injuries["report_status"].isin(["Out", "Questionable", "Doubtful"])
            ]
        else:
            # If no status column, just count all recent injuries
            serious_injuries = team_injuries

        count = len(serious_injuries)
        
        # Sanity check: cap at 15 (no team has more than 15 serious injuries)
        if count > 15:
            return 15
        
        return count

    def _calculate_team_features(self, team, team_history, current_date, historical_games, prefix="team"):
        """Calculate comprehensive features for one team"""
        features = {}

        recent_games = team_history.tail(self.lookback_games)

        if len(recent_games) == 0:
            return None

        # Feature 1: Win Rate (last N games)
        wins = sum(
            1 for _, game in recent_games.iterrows()
            if (game["home_team"] == team and game["home_score"] > game["away_score"]) or
               (game["away_team"] == team and game["away_score"] > game["home_score"])
        )
        features[f"{prefix}_win_rate"] = wins / len(recent_games)

        # Feature 2: Average Points Scored
        points_scored = [
            int(game["home_score"]) if game["home_team"] == team else int(game["away_score"])
            for _, game in recent_games.iterrows()
        ]
        features[f"{prefix}_avg_points_scored"] = np.mean(points_scored)

        # Feature 3: Average Points Allowed
        points_allowed = [
            int(game["away_score"]) if game["home_team"] == team else int(game["home_score"])
            for _, game in recent_games.iterrows()
        ]
        features[f"{prefix}_avg_points_allowed"] = np.mean(points_allowed)

        # Feature 4: Point Differential
        features[f"{prefix}_point_diff"] = features[f"{prefix}_avg_points_scored"] - features[f"{prefix}_avg_points_allowed"]

        # Feature 5: Rest Days
        last_game = team_history.iloc[-1]
        last_game_date = pd.to_datetime(last_game["gameday"])
        current_date = pd.to_datetime(current_date)
        rest_days = (current_date - last_game_date).days
        
        # CRITICAL FIX: Cap rest days at 21 (3 weeks) for sanity
        # If more than 21, it means we're predicting with stale data
        if rest_days > 21:
            rest_days = 21  # Treat as maximum rest
        
        features[f"{prefix}_rest_days"] = rest_days

        # Feature 6: Bye Week
        features[f"{prefix}_had_bye"] = 1 if 7 <= rest_days <= 14 else 0

        # Feature 7: Short Rest
        features[f"{prefix}_short_rest"] = 1 if rest_days < 6 else 0

        # Feature 8: Is Home Team
        features[f"{prefix}_is_home"] = 1 if prefix == "home" else 0

        # Feature 9: Injury Count (FIXED)
        injury_count = self._get_injury_count(team, current_date)
        features[f"{prefix}_injury_count"] = injury_count

        # Feature 10: Weighted Recent Form
        very_recent = recent_games.tail(self.WIN_RATE_LOOKBACK)
        recent_results = [
            1 if (game["home_team"] == team and game["home_score"] > game["away_score"]) or
                 (game["away_team"] == team and game["away_score"] > game["home_score"])
            else 0
            for _, game in very_recent.iterrows()
        ]
        
        if len(recent_results) >= 3:
            weights = np.array([0.2, 0.3, 0.5])[-len(recent_results):]
            weights = weights / weights.sum()
            features[f"{prefix}_weighted_form"] = np.average(recent_results, weights=weights)
        else:
            features[f"{prefix}_weighted_form"] = np.mean(recent_results) if recent_results else 0.0

        # Feature 11: Strength of Schedule
        features[f"{prefix}_strength_of_schedule"] = self._calculate_strength_of_schedule(
            recent_games, historical_games
        )

        # Feature 12 & 13: Home/Away Performance Splits
        home_away_splits = self._calculate_home_away_splits(team_history, team)
        features[f"{prefix}_home_win_rate"] = home_away_splits['home_win_rate']
        features[f"{prefix}_away_win_rate"] = home_away_splits['away_win_rate']

        # Feature 14: Scoring Trend
        features[f"{prefix}_scoring_trend"] = self._calculate_scoring_trend(recent_games, team)

        return features

    def _calculate_strength_of_schedule(self, recent_games, historical_games):
        """Calculate average win rate of recent opponents"""
        opponent_win_rates = []
        
        for _, game in recent_games.iterrows():
            opponent = game["away_team"] if game["home_team"] != game["away_team"] else game["home_team"]
            
            # Get opponent's history up to this game
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
                opp_win_rate = opp_wins / len(opponent_games)
                opponent_win_rates.append(opp_win_rate)
        
        return np.mean(opponent_win_rates) if opponent_win_rates else 0.5

    def _calculate_home_away_splits(self, team_history, team):
        """Calculate separate win rates for home and away games"""
        home_games = team_history[team_history["home_team"] == team]
        away_games = team_history[team_history["away_team"] == team]
        
        home_wins = sum(1 for _, g in home_games.iterrows() if g["home_win"] == 1)
        away_wins = sum(1 for _, g in away_games.iterrows() if g["home_win"] == 0)
        
        home_win_rate = home_wins / len(home_games) if len(home_games) > 0 else 0.5
        away_win_rate = away_wins / len(away_games) if len(away_games) > 0 else 0.5
        
        return {
            'home_win_rate': home_win_rate,
            'away_win_rate': away_win_rate
        }

    def _calculate_scoring_trend(self, recent_games, team):
        """Calculate if team's scoring is trending up or down"""
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

    def _calculate_matchup_features(self, home_team, away_team, historical_games, current_date):
        """Calculate features specific to this matchup"""
        features = {}
        
        # Head-to-head history
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
        
        return features

    def create_prediction_features(self, home_team, away_team, current_date=None):
        """Create features for a future game prediction"""
        if current_date is None:
            current_date = datetime.now()

        current_date = pd.to_datetime(current_date)
        
        # Normalize team names
        home_team = self.normalize_team_name(home_team)
        away_team = self.normalize_team_name(away_team)

        # Get all historical games before prediction date
        historical_games = self.schedules[self.schedules["gameday"] < current_date]

        # Get team histories
        home_history = self._get_team_history(historical_games, home_team)
        away_history = self._get_team_history(historical_games, away_team)

        # Better error messages
        available_teams = sorted(self.schedules['home_team'].unique())
        
        if len(home_history) < self.MIN_GAMES_REQUIRED:
            raise ValueError(
                f"Not enough historical data for '{home_team}'. Need at least {self.MIN_GAMES_REQUIRED} games.\n"
                f"Available teams in data: {', '.join(available_teams[:10])}..."
            )
        if len(away_history) < self.MIN_GAMES_REQUIRED:
            raise ValueError(
                f"Not enough historical data for '{away_team}'. Need at least {self.MIN_GAMES_REQUIRED} games.\n"
                f"Available teams in data: {', '.join(available_teams[:10])}..."
            )

        # Check data staleness
        last_home_game = home_history.iloc[-1]['gameday']
        last_away_game = away_history.iloc[-1]['gameday']
        days_since_home = (current_date - last_home_game).days
        days_since_away = (current_date - last_away_game).days
        
        if days_since_home > 60 or days_since_away > 60:
            print(f"\n  ⚠ WARNING: Using stale data!")
            print(f"    {home_team} last game: {days_since_home} days ago")
            print(f"    {away_team} last game: {days_since_away} days ago")
            print(f"    Predictions may be unreliable.\n")

        # Calculate features
        home_features = self._calculate_team_features(
            home_team, home_history, current_date, historical_games, prefix="home"
        )
        away_features = self._calculate_team_features(
            away_team, away_history, current_date, historical_games, prefix="away"
        )

        if home_features is None or away_features is None:
            raise ValueError("Failed to calculate features for one or both teams.")

        # Combine features
        features = {**home_features, **away_features}
        
        # Add matchup features
        matchup_features = self._calculate_matchup_features(
            home_team, away_team, historical_games, current_date
        )
        features.update(matchup_features)

        # Add metadata
        features["home_team"] = home_team
        features["away_team"] = away_team

        return features