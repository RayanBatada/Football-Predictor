'''
from sys import prefix
import pandas as pd
import numpy as np
from datetime import datetime


class NFLFeatureEngineer:
    """
    Engineer features for NFL game prediction

    1. For each game, create features for BOTH the home and away teams.
    2. Features describe recent performance (wins, points, rest, injuries)
    3. Target variable: Did the home team win? (1 = yes,  0 = no)
    """
    
    WIN_RATE_LOOKBACK = 3 # number of recent games to consider for win rate
    MIN_GAMES_REQUIRED = 3 # minimum number of games required to create features


    def __init__(self, lookback_games=5):
        self.lookback_games = lookback_games # default 5
        self.schedules = None
        self.injuries = None
        self.weekly_stats = None


    def load_data(self, schedules_df, injuries_df, weekly_stats_df=None): # returns an updated instance of the class (self)
        self.schedules = schedules_df.copy()
        self.injuries = injuries_df.copy() 
        self.weekly_stats = weekly_stats_df.copy() if weekly_stats_df is not None else None

        # Sort by date to ensure chronological order (so we don't train model on games that haven't happened yet)
        self.schedules['gameday'] = pd.to_datetime(self.schedules['gameday'])
        self.schedules = self.schedules.sort_values('gameday').reset_index(drop=True) # reset_index resets the index --> TODO: determine if necessary

        if self.weekly_stats is not None and len(self.weekly_stats) > 0:
            team_col = None
            for possible_name in ['team', 'recent_team', 'team_abbr', 'posteam']:
                if possible_name in self.weekly_stats.columns:
                    team_col = possible_name
                    break

            if team_col is not None and {"season", "week"}.issubset(self.weekly_stats.columns):
                self.weekly_stats = self.weekly_stats.rename(columns={team_col: 'team'})
            else:
                self.weekly_stats = None


        self.schedules["home_win"] = (self.schedules["home_score"] > self.schedules["away_score"]).astype(int)  # 1 if home team wins, 0 otherwise


        return self


    def create_features(self):
        """
        1. Loop through every game in self.schedules
        2. For each game, call _create_game_features(game, idx) to create features for that game
        3. Collect all feature dictionaries into a list
        4. Convert list back to a DataFrame
        5. Return the DataFrame of features (print out stats about the features created)
        """

        features_list = []
        for idx, game in self.schedules.iterrows():
            if idx % 100 == 0:
                print(f"Processing game {idx} / {len(self.schedules)}")

            game_features = self._create_game_features(game, idx)

            if game_features is not None:
                features_list.append(game_features)

        features_df = pd.DataFrame(features_list) # convert collected features into a DataFrame

        print("Feature engineering complete.")
        print(f"Created features for {len(features_df)} games.")
        print(f"Feature columns: {features_df.columns.tolist()}")


        return features_df

    
    def _create_game_features(self, game, game_idx):
        """
        game: A row from the schedules DataFrame (the game to predict)
        game_idx: Row Index (position in chronological order)

        Calculate statistics about both teams based on recent performance BEFORE current game
        """

        home_team = game["home_team"]
        away_team = game["away_team"]
        game_date = game["gameday"]

        historical_games = self.schedules.iloc[:game_idx]
    
        home_history = self._get_team_history(historical_games, home_team)
        away_history = self._get_team_history(historical_games, away_team)

        if len(home_history) < NFLFeatureEngineer.MIN_GAMES_REQUIRED or len(away_history) < NFLFeatureEngineer.MIN_GAMES_REQUIRED:
            return None
    

        home_features = self._calculate_team_features(home_team, home_history, game_date, game["season"], game["week"], prefix="home")
        away_features = self._calculate_team_features(away_team, away_history, game_date, game["season"], game["week"], prefix="away")

        if home_features is None or away_features is None:
            return None
        
        diff_features = {
            "diff_win_rate": home_features["home_win_rate"] - away_features["away_win_rate"],
            "diff_avg_points_scored": home_features["home_avg_points_scored"] - away_features["away_avg_points_scored"],
            "diff_avg_points_allowed": home_features["home_avg_points_allowed"] - away_features["away_avg_points_allowed"],
            "diff_point_diff": home_features["home_point_diff"] - away_features["away_point_diff"],
            "diff_rest_days": home_features["home_rest_days"] - away_features["away_rest_days"],
            "diff_injury_count": home_features["home_injury_count"] - away_features["away_injury_count"],
            "diff_recent_form": home_features["home_recent_form"] - away_features["away_recent_form"],
        }

        features = {**home_features, **away_features, **diff_features} # the ** means to unpack the dictionaries
        
        # Add metadata to help debug and track predictions
        features["game_id"] = game.get("game_id", f"{game_date}_{home_team}_vs_{away_team}")
        features["season"] = game["season"]
        features["week"] = game["week"]
        features["home_team"] = home_team
        features["away_team"] = away_team

        features["target"] = game["home_win"]  # Target variable

        return features
        

    def _get_team_history(self, historical_games, team):
        """
        Get all games a team played (either home or away) from historical_games DataFrame
        Returns DataFrame of all games this team played, sorted by date
        """

        team_games = historical_games[
            (historical_games["home_team"] == team) | (historical_games["away_team"] == team)
        ].copy()

        team_games = team_games.sort_values("gameday").reset_index(drop=True)

        return team_games


    def _get_injury_count(self, team, game_date):
        """
        Find the injury data for "team", and then calculate all those injuries that have have "Out" or "Questionable" status
        """

        if self.injuries is None or len(self.injuries) == 0:
            return 0

        game_date = pd.to_datetime(game_date)

        # Filter injuries for the team with serious status
        team_injuries = self.injuries[self.injuries["team"] == team]

        serious_injuries = team_injuries[
            team_injuries["report_status"].isin(["Out", "Questionable"])
        ]

        return len(serious_injuries)


    def _calculate_team_features(self, team, team_history, current_date=None, season=None, week=None, prefix="team"):
        """
        Calculate ALL features for ONE team

        {prefix}_win_rate: Wins/ total games (last N games)
        {prefix}_avg_points_scored: Mean points scored
        {prefix}_avg_points_allowed: Mean points given up
        {prefix}_point_diff: Point differential (scored - allowed)
        {prefix}_rest_days: Days since last game
        {prefix}_had_bye: 1 if had a bye week
        {prefix}_short_rest: 1 if <6 days rest
        {prefix}_home: 1 if home game
        {prefix}_injury_count: Number of injuries
        {prefix}_recent_form: List of last 3 game results (1=win, 0=loss), 1.0 is 3-0
        """

        features = {}

        recent_games = team_history.tail(self.lookback_games)

        # Feature 1: Win Rate
        wins = 0
        for _, game in recent_games.iterrows():
            if game["home_team"] == team: # if home team
                if game["home_score"] > game["away_score"]:
                    wins += 1 # if home team won
            else:
                if game["away_score"] > game["home_score"]:
                    wins += 1 # if away team won

        features[f"{prefix}_win_rate"] = wins / len(recent_games) if len(recent_games) > 0 else 0.0


        # Feature 2: Average Points Scored
        points_scored = 0
        for _, game in recent_games.iterrows():
            # TODO: verify this works with pandas types
            if game["home_team"] == team:
                points_scored += int(game["home_score"])
            else:
                points_scored += int(game["away_score"])

        features[f"{prefix}_avg_points_scored"] = points_scored / len(recent_games) if len(recent_games) > 0 else 0.0


        # Feature 3: Average Points Allowed
        points_allowed = 0
        for _, game in recent_games.iterrows():
            if game["home_team"] == team:
                points_allowed += int(game["away_score"])
            else:
                points_allowed += int(game["home_score"])

        features[f"{prefix}_avg_points_allowed"] = points_allowed / len(recent_games) if len(recent_games) > 0 else 0.0

        # Feature 4: Point Differential
        features[f"{prefix}_point_diff"] = features[f"{prefix}_avg_points_scored"] - features[f"{prefix}_avg_points_allowed"]



        # Feature 5: Rest Days
        last_game = team_history.iloc[-1] # iloc is used to access rows by integer location
        last_game_date = pd.to_datetime(last_game["gameday"]) # must use pd.to_datetime to convert to dateTime type
        current_date = pd.to_datetime(current_date)
        rest_days = (current_date - last_game_date).days # .days gives the difference in days because data type is dateTime
        features[f"{prefix}_rest_days"] = rest_days 

        # Feature 6: Bye Week
        features[f"{prefix}_had_bye"] = 1 if 7 <= rest_days <= 14 else 0


        # Feature 7: Short Rest
        features[f"{prefix}_short_rest"] = 1 if rest_days < 6 else 0


        # Feature 8: Home/Away
        features[f"{prefix}_home"] = 1 if prefix == "home" else 0

        # Feature 9: Injury Count
        if self.injuries is not None:
            injury_count = self._get_injury_count(team, current_date)

        features[f"{prefix}_injury_count"] = injury_count if self.injuries is not None else 0


        # Feature 10: Recent Form
        

        very_recent = recent_games.tail(self.WIN_RATE_LOOKBACK)
        recent_wins = 0

        for _, game in very_recent.iterrows():
            if game["home_team"] == team:
                if game["home_score"] > game["away_score"]:
                    recent_wins += 1
            else:
                if game["away_score"] > game["home_score"]:
                    recent_wins += 1

        features[f"{prefix}_recent_form"] = recent_wins / len(very_recent) if len(very_recent) > 0 else 0.0

        if self.weekly_stats is not None and season is not None and week is not None:
            team_weekly = self.weekly_stats[
                (self.weekly_stats["team"] == team)
                & (self.weekly_stats["season"] == season)
                & (self.weekly_stats["week"] < week)
            ]

            if len(team_weekly) > 0:
                team_weekly = team_weekly.sort_values("week").tail(self.lookback_games)
                weekly_columns = [
                    "points_for",
                    "points_against",
                    "total_yards",
                    "offense_yards",
                    "passing_yards",
                    "rushing_yards",
                    "turnovers",
                    "takeaways",
                    "sacks",
                    "passing_epa",
                    "rushing_epa",
                    "epa",
                ]

                available_weekly = [col for col in weekly_columns if col in team_weekly.columns]
                for col in available_weekly:
                    features[f"{prefix}_weekly_{col}_avg"] = team_weekly[col].mean()

        return features



    # TODO: Define for when creating features for a future game (not in schedules)
    def create_prediction_features(self,  home_team, away_team, current_date=None):
        """
        Create features for a future game between home_team and away_team on current_date

        Returns a dictionary of features
        """

        if current_date is None:
            current_date = datetime.now()

        current_date = pd.to_datetime(current_date)

        historical_games = self.schedules # get all historical football games

        # Get historical games of home and away teams 
        home_history = self._get_team_history(historical_games, home_team)
        away_history = self._get_team_history(historical_games, away_team) 

        # Get history of home and away teams before current_date:
        home_history = home_history[home_history["gameday"] < current_date]
        away_history = away_history[away_history["gameday"] < current_date]

        # Check if enough historical data exists, otherwise raise error stating not enough historical data to make a prediction
        if len(home_history) < NFLFeatureEngineer.MIN_GAMES_REQUIRED or len(away_history) < NFLFeatureEngineer.MIN_GAMES_REQUIRED:
            raise ValueError("Not enough historical data to create features for one or both teams.")

        # Calculate features for both teams
        home_features = self._calculate_team_features(home_team, home_history, current_date, prefix="home")
        away_features = self._calculate_team_features(away_team, away_history, current_date, prefix="away")

        # More Error Handling:
        if home_features is None or away_features is None:
            raise ValueError("Failed to calculate features for one or both teams.")

        # Organize and Finalize Feature Data for Processing
        features = {**home_features, **away_features} # merge both dictionaries

        features["home_team"] = home_team # add metadata
        features["away_team"] = away_team # add metadata

        return features
'''
import pandas as pd

from datetime import datetime


class NFLFeatureEngineer:
    """ Engineer features for NFL game prediction

    1. For each game, create features for BOTH the home and away team.
    2. Features describe recent performance (wins, points, rest, injuries)
    3. Target variable: Did the home team win? (1 = yes, 0 = no)
    """
    WIN_RATE_LOOKBACK = 3  # Number of recent games to consider for win rate
    MIN_GAMES_REQUIRED = 3  # Minimum games required to generate features
    
    def __init__(self, lookback_games=5):
        self.lookback_games = lookback_games  # Number of recent games to consider for features
        self.schedules = None
        self.injuries = None
        self.weekly_stats = None

    def load_data(self, schedules_df, injuries_df, weekly_stats_df=None):
        self.schedules = schedules_df.copy()
        self.injuries = injuries_df.copy()
        self.weekly_stats = weekly_stats_df.copy() if weekly_stats_df is not None else None

        # Sort by date to ensure chronological order (so we don't train model on games that haven't happened yet)
        self.schedules["gameday"] = pd.to_datetime(self.schedules["gameday"])
        self.schedule = self.schedules.sort_values("gameday").reset_index(drop=True) # TODO: Determine if need

        if self.weekly_stats is not None and len(self.weekly_stats) > 0:
            team_col = None
            for possible_name in ["team", "recent_team", "team_abbr", "posteam"]:
                if possible_name in self.weekly_stats.columns:
                    team_col = possible_name
                    break

            if team_col is not None and {"season", "week"}.issubset(self.weekly_stats.columns):
                self.weekly_stats = self.weekly_stats.rename(columns={team_col: "team"})
            else:
                self.weekly_stats = None

        self.schedules["home_win"] = (
            self.schedules["home_score"] > self.schedules["away_score"]
        ).astype(int)

        return self
    
    def create_features(self):
        """
        1. Loop through every game in self.schedules
        2. For each game, call _create_game_features(game, idx)
        3. Collect all feature dictionaries into a list
        4. Convert list back to a DataFrame
        5. Return DataFrame of features (print out stats)
        """
        features_list = []

        for idx, game in self.schedules.iterrows():
            if idx % 100 == 0:
                print(f"Processing game {idx}/{len(self.schedules)}...")
            
            game_features = self._create_game_features(game, idx)

            if game_features is not None:
                features_list.append(game_features)
            
        features_df = pd.DataFrame(features_list)

        # Print out some stats about the features
        print("Feature engineering complete.")
        print(f"Generated {len(features_df)} feature rows.")
        print(f"Columns: {features_df.columns.tolist()}")

        return features_df
    
    
    def _create_game_features(self, game, game_idx):
        """
        game: A row from schedules DataFrame (the game to predict)
        game_idx: Row index (position in choronological order)

        Calculate statistics about both teams based on recent performance BEFORE current game
        """
        home_team = game["home_team"]
        away_team = game["away_team"]
        game_date = game["gameday"]

        historical_games = self.schedules.iloc[:game_idx]

        home_history = self._get_team_history(historical_games, home_team)
        away_history = self._get_team_history(historical_games, away_team)

        if len(home_history) < self.MIN_GAMES_REQUIRED or len(away_history) < self.MIN_GAMES_REQUIRED:
            return None # Skip this game, not enough data
        
        home_features = self._calculate_team_features(
            home_team,
            home_history,
            game_date,
            season=game["season"],
            week=game["week"],
            prefix="home",
        )
        away_features = self._calculate_team_features(
            away_team,
            away_history,
            game_date,
            season=game["season"],
            week=game["week"],
            prefix="away",
        )

        diff_features = {
            "diff_win_rate": home_features["home_win_rate"] - away_features["away_win_rate"],
            "diff_avg_points_scored": home_features["home_avg_points_scored"] - away_features["away_avg_points_scored"],
            "diff_avg_points_allowed": home_features["home_avg_points_allowed"] - away_features["away_avg_points_allowed"],
            "diff_point_diff": home_features["home_point_diff"] - away_features["away_point_diff"],
            "diff_rest_days": home_features["home_rest_days"] - away_features["away_rest_days"],
            "diff_injury_count": home_features["home_injury_count"] - away_features["away_injury_count"],
            "diff_recent_form": home_features["home_recent_form"] - away_features["away_recent_form"],
        }

        features = {**home_features, **away_features, **diff_features}
        
        # Add metadata to help debug and track predictions
        features["game_id"] = game.get("game_id", f"{game_date}_{home_team}_vs_{away_team}")
        features["season"] = game["season"]
        features["week"] = game["week"]
        features["home_team"] = home_team
        features["away_team"] = away_team

        features["target"] = game["home_win"]

        return features
    
    def _get_team_history(self, historical_games, team):
        """
        Get all games a team played (either home or away)
        Returns DataFrame of all games this team played, sorted by date
        """
        team_games = historical_games[
            (historical_games["home_team"] == team) | (historical_games["away_team"] == team)
        ].sort_values("gameday").reset_index(drop=True)
        
        return team_games
    
    def _get_injury_count(self, team, game_date):
        """
        Find the injury data for "team", and then calculate all those injuries that have "Out" or "Questionable" status
        """
        if self.injuries is None or len(self.injuries) == 0:
            return 0
        
        game_date = pd.to_datetime(game_date)
        
        team_injuries = self.injuries[self.injuries["team"] == team]

        serious_injuries = team_injuries[
            team_injuries["report_status"].isin(["Out", "Questionable"])
        ]

        return len(serious_injuries)

    def _calculate_team_features(self, team, team_history, current_date, season=None, week=None, prefix="team"):
        """
        Calculate ALL features for ONE team
        
        {prefix}_win_rate: Wins / total games (last N)
        {prefix}_avg_points_scored: Mean points scored
        {prefix}_avg_points_allowed: Mean points given up
        {prefix}_point_diff: Point differential
        {prefix}_rest_days: Days since last game
        {prefix}_had_bye: 1 if had a bye week
        {prefix}_short_rest: 1 if <6 days rest
        {prefix}_home: 1 if home game
        {prefix}_injury_count: Number of injuries
        {prefix}_recent_form: List of last 3 game results (1=win,0=loss), 1.0 if 3-0
        """
        features = {}
        
        recent_games = team_history.tail(self.lookback_games)

        # FEATURE 1: Win Rate
        wins = 0
        for _, game in recent_games.iterrows():
            if game["home_team"] == team:
                if game["home_score"] > game["away_score"]:
                    wins += 1
            else:
                if game["away_score"] > game["home_score"]:
                    wins += 1
        features[f"{prefix}_win_rate"] = wins / len(recent_games) if len(recent_games) > 0 else 0.0
        
        # FEATURE 2: Average Points Scored
        points_scored = 0
        # TODO: TEACHING - SESSION - RAYAN - CHANGES MADE HERE
        for _, game in recent_games.iterrows():
            if game["home_team"] == team:
                points_scored += int(game["home_score"])
            else:
                points_scored += int(game["away_score"])
        features[f"{prefix}_avg_points_scored"] = points_scored / len(recent_games) if len(recent_games) > 0 else 0.0

        # FEATURE 3: Average Points Allowed
        points_allowed = 0
        for _, game in recent_games.iterrows():
            if game["home_team"] == team:
                points_allowed += int(game["away_score"])
            else:
                points_allowed += int(game["home_score"])
        features[f"{prefix}_avg_points_allowed"] = points_allowed / len(recent_games) if len(recent_games) > 0 else 0.0

        # FEATURE 4: Point Differential
        features[f"{prefix}_point_diff"] = features[f"{prefix}_avg_points_scored"] - features[f"{prefix}_avg_points_allowed"]

        # FEATURE 5: Rest Days
        last_game = team_history.iloc[-1]
        last_game_date = pd.to_datetime(last_game["gameday"])
        current_date = pd.to_datetime(current_date)
        rest_days = (current_date - last_game_date).days
        features[f"{prefix}_rest_days"] = rest_days
        
        # FEATURE 6: Had Bye Week
        features[f"{prefix}_had_bye"] = 1 if 7 <= rest_days <= 14 else 0

        # FEATURE 7: Short Rest
        features[f"{prefix}_short_rest"] = 1 if rest_days < 6 else 0

        # FEATURE 8: Is Home Game
        features[f"{prefix}_home"] = 1 if prefix == "home" else 0

        # FEATURE 9: Injury Count
        if self.injuries is not None:
            injury_count = self._get_injury_count(team, current_date)
        features[f"{prefix}_injury_count"] = injury_count if self.injuries is not None else 0

        # FEATURE 10: Recent Form
        very_recent = recent_games.tail(self.WIN_RATE_LOOKBACK)
        recent_wins = 0
        for _, game in very_recent.iterrows():
            if game["home_team"] == team:
                if game["home_score"] > game["away_score"]:
                    recent_wins += 1
            else:
                if game["away_score"] > game["home_score"]:
                    recent_wins += 1
        features[f"{prefix}_recent_form"] = recent_wins / len(very_recent) if len(very_recent) > 0 else 0

        if self.weekly_stats is not None and season is not None and week is not None:
            team_weekly = self.weekly_stats[
                (self.weekly_stats["team"] == team)
                & (self.weekly_stats["season"] == season)
                & (self.weekly_stats["week"] < week)
            ]

            if len(team_weekly) > 0:
                team_weekly = team_weekly.sort_values("week").tail(self.lookback_games)
                weekly_columns = [
                    "points_for",
                    "points_against",
                    "total_yards",
                    "offense_yards",
                    "passing_yards",
                    "rushing_yards",
                    "turnovers",
                    "takeaways",
                    "sacks",
                    "passing_epa",
                    "rushing_epa",
                    "epa",
                ]

                available_weekly = [col for col in weekly_columns if col in team_weekly.columns]
                for col in available_weekly:
                    features[f"{prefix}_weekly_{col}_avg"] = team_weekly[col].mean()

        return features
    


    # TODO: Define for when creating features for a future game
    def create_prediction_features(self, home_team, away_team, current_date=None):
        
        if current_date is None:
            current_date = datetime.now()
            # pd.Timestamp.now() --> TODO: Verify this will work as well
        
        current_date = pd.to_datetime(current_date)

        # 1. Get Team History
        home_history = self._get_team_history(self.schedules, home_team)
        away_history = self._get_team_history(self.schedules, away_team)

        home_history = home_history[home_history["gameday"] < current_date]
        away_history = away_history[away_history["gameday"] < current_date]

        if len(home_history) < 3 or len(away_history) < 3:
            raise ValueError("Not enough historical data to create prediction features for the specified teams.")

        # 2. Calculate Team Features
        home_features = self._calculate_team_features(home_team, home_history, current_date, prefix="home")
        away_features = self._calculate_team_features(away_team, away_history, current_date, prefix="away")

        # 3. Organize and Finalize Feature Data for Processing
        diff_features = {
            "diff_win_rate": home_features["home_win_rate"] - away_features["away_win_rate"],
            "diff_avg_points_scored": home_features["home_avg_points_scored"] - away_features["away_avg_points_scored"],
            "diff_avg_points_allowed": home_features["home_avg_points_allowed"] - away_features["away_avg_points_allowed"],
            "diff_point_diff": home_features["home_point_diff"] - away_features["away_point_diff"],
            "diff_rest_days": home_features["home_rest_days"] - away_features["away_rest_days"],
            "diff_injury_count": home_features["home_injury_count"] - away_features["away_injury_count"],
            "diff_recent_form": home_features["home_recent_form"] - away_features["away_recent_form"],
        }

        features = {**home_features, **away_features, **diff_features}
        
        features["home_team"] = home_team
        features["away_team"] = away_team

        return features
