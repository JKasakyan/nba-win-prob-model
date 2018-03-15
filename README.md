# nba-win-prob-model
Machine learning model to predict single game win probability for home team in NBA game

## Cloning the environment
Use the provided environment.yml file to clone the project environment

## Usage
**To predict the home team win probability for a given NBA game, activate the cloned environment and run:**

*python single_game_prediction.py home_team away_team game_date boxscores_csv_path season_summaries_csv_path*

  **WARNING: In the current state of the script, predictions can only be made on games between the 1984 and 2017 NBA seasons if using the provided CSVs ([nba_boxscores_1984_2018.csv](/Data/nba_boxscores_1984_2018.csv) and [nba_season_summaries_1984_2018.csv](/Data/nba_season_summaries_1984_2018.csv))**

Arguments:
  * home_team - 3 letter abbreviation for home team (basketball-reference boxscore initials)
  * away_team - 3 letter abbreviation for away team (basketball-reference boxscore initials)
  * game_date - the date of the game to predict (format=YYYY-MM-DD)
  * (optional) boxscores_csv_path: string, path to boxscores csv, see nba_boxscores_1984_2018.csv for format
  * (optional) season_summaries_csv_path: string, path to season summaries csv, see nba_season_summaries_1984_2018.csv for format
  
  **WARNING: if boxscores_csv_path or season_summaries_csv_path are not provided, the script will attempt to scrape the needed data. This requires an internet connection and will take time. THIS FEATURE IS CURRENTLY UNTESTED AND MAY NOT FUNCTION PROPERLY!**

## Example
python single_game_prediction.py MIA CHO 2016-10-28 ../Data/nba_boxscores_1984_2018.csv ../Data/nba_season_summaries_1984_2018.csv
