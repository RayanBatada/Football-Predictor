
import pandas as pd
import nfl_data_py as nfl

START_YEAR = 2023
END_YEAR = 2024


def download_nfl_data(years=list(range(START_YEAR, END_YEAR + 1))):
    # print for each one as well

    # Import schedules
    schedules_df = nfl.import_schedules(years)
    print("Schedules imported: ", schedules_df.shape)

    # Import seasonal player stats
    player_stats_df = nfl.import_seasonal_data(years)
    print("Player stats imported: ", player_stats_df.shape)

    # Import weekly player stats
    weekly_player_stats_df = nfl.import_weekly_data(years)
    print("Weekly player stats imported: ", weekly_player_stats_df.shape)

    # Import injuries
    injuries_df = nfl.import_injuries(years)
    print("Injuries imported: ", injuries_df.shape)

    return schedules_df, player_stats_df, weekly_player_stats_df, injuries_df



if __name__ == "__main__":
    data = download_nfl_data()
