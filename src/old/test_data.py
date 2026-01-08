
import nfl_data_py as nfl
#import nflreadpy as nflrp
#import pandas as pd


games_2023 = nfl.import_schedules([2023])

rosters = nfl.import_rosters([2023])

draft_picks = nfl.import_draft_picks([2023])






#pbp_current_season = nfl.import_pbp_data([2024])
#print(pbp_current_season.head())


#player_game_stats_2023  = nflrp.load_player_stats([2023])
#print(player_game_stats_2023.head())

#print(player_stats.shape)
#print(player_stats.columns[:15])
#print(player_stats.head(3))

'''

print(games_2024.head())
print(f"Total games in 2024 season: {len(games_2024)}")
print("Raw data preview:")
print(games_2024)


print("Feature Data: ", games_2024[['week', 'gameday', 'home_team', 'away_team', 'home_score', 'away_score', "location"]].head())

print("Players 2024 Season:") 
players_2024 = nfl.import_players(seasons=[2024])
print(players_2024.head())
print(f"Total players in 2024 season: {len(players_2024)}")
print("Raw player data preview:")
print(players_2024)

'''