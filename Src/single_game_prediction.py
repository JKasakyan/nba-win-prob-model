'''
USAGE
arguments:
  home_team - 3 letter abbreviation for home team (basketball-reference boxscore initials)
  away_team - 3 letter abbreviation for away team (basketball-reference boxscore initials)
  game date - the date of the game to predict (format=YYYY-MM-DD)
  (optional) boxscores_csv_path: string, path to boxscores csv, see nba_boxscores_1984_2018.csv for format
  (optional) season_summaries_csv_path: string, path to season summaries csv, see nba_season_summaries_1984_2018.csv for format

  WARNING: if boxscores_csv_path or season_summaries_csv_path are not provided, the script will attempt to scrape the needed data. This requires an internet connection and will take time. THIS FEATURE IS CURRENTLY UNTESTED AND MAY NOT FUNCTION PROPERLY!

ex: python single_game_prediction.py MIA CHO 2016-10-28 ../Data/nba_boxscores_1984_2018.csv ../Data/nba_season_summaries_1984_2018.csv
'''

import os
import sys
import datetime
import subprocess
from subprocess import Popen, PIPE
import glob

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
from sklearn.externals import joblib

nba_1984_2018_initials = {
    'Atlanta Hawks': 'ATL',
    'Boston Celtics': 'BOS',
    'Brooklyn Nets': 'BRK',
    'Charlotte Hornets': 'CHH',
    'Charlotte Bobcats': 'CHO',
    'Chicago Bulls': 'CHI',
    'Cleveland Cavaliers': 'CLE',
    'Dallas Mavericks': 'DAL',
    'Denver Nuggets': 'DEN',
    'Detroit Pistons': 'DET',
    'Golden State Warriors': 'GSW',
    'Houston Rockets': 'HOU',
    'Indiana Pacers': 'IND',
    'Kansas City Kings': 'KCK',
    'Los Angeles Clippers': 'LAC',
    'Los Angeles Lakers': 'LAL',
    'Memphis Grizzlies': 'MEM',
    'Miami Heat': 'MIA',
    'Milwaukee Bucks': 'MIL',
    'Minnesota Timberwolves': 'MIN',
    'New Orleans Hornets' : 'NOP',
    'New Orleans/Oklahoma City Hornets': 'NOK',
    'New Orleans Pelicans': 'NOP',
    'New Jersey Nets': 'NJN',
    'New York Knicks': 'NYK',
    'Oklahoma City Thunder': 'OKC',
    'Orlando Magic': 'ORL',
    'Philadelphia 76ers': 'PHI',
    'Phoenix Suns': 'PHO',
    'Portland Trail Blazers': 'POR',
    'Sacramento Kings': 'SAC',
    'San Diego Clippers': 'SDC',
    'San Antonio Spurs': 'SAS',
    'Seattle SuperSonics': 'SEA',
    'Toronto Raptors': 'TOR',
    'Vancouver Grizzlies': 'VAN',
    'Utah Jazz': 'UTA',
    'Washington Bullets': 'WSB',
    'Washington Wizards': 'WAS'
}
nba_1984_2018_initials_reversed = {v:k for k, v in nba_1984_2018_initials.items()}

boxscore_season_range_mask = lambda df, start_year, end_year:  (df["season"] >= start_year) & (df["season"] <= end_year)
boxscore_date_range_mask = lambda df, start_date, end_date: (df["date"] >= start_date) & (df["date"] <= end_date)
boxscore_team_mask = lambda df, team_initials: (df["team1"] == team_initials) | (df["team2"] == team_initials)
boxscore_regular_season_mask = lambda df: pd.isnull(df["playoff"])

summary_season_range = lambda df, start_year, end_year: df.loc[start_year:end_year]
summary_season_query = lambda df, years, teams, col_names: df.loc[(years, teams), col_names]
summary_season_remove_league_average = lambda: df

# Helper functions
def margin_for_team(abbrev, margins):
    '''
    find the average margin of victory for a team over a period of time

    abbrev: string, 3 letter initial of NBA team
    margins: dict, key is team initials and value is list containing the margin of victory for each game

    returns average margin of victory for given team over time period encompassed by margins.
    '''
    return sum(margins[abbrev]) / len(margins[abbrev])

def weighted_margin_for_team(abbrev, margins):
    '''
    find the weighted average margin of victory for a team, where previous season average margin is weighted by ((82 - games played) / 82) and current season average margin is weighted by (games played / 82)

    abbrev: string, 3 letter initial of NBA team
    margins: dict, key is team initials and value is list containing the margin of victory for each game where first element equals last season average margin of victory

    returns weighted average margin of victory for given team over time period encompassed by margins.
    '''
    last_season_margin = margins[abbrev][0]
    this_season_margins = margins[abbrev][1:]
    gp = len(this_season_margins)
    if gp == 0:
        # no games played, return last season's margin
        return last_season_margin
    elif gp > 82:
        # team in playoffs, use only this season's margin
        return sum(this_season_margins) / len(this_season_margins)
    else:
        current_season_margin = sum(this_season_margins) / len(this_season_margins)
        return (gp / 82 * current_season_margin) + (((82 - gp) / 82)  * last_season_margin)

def sos_for_team(abbrev, schedule, margins):
    '''
    find the strength of schedule for a team over a period of time, where sos is defined as average margin of victory of opponents faced so far, weighted by games played.

    abbrev: string, 3 letter initial of NBA team
    margins: dict, key is team initials and value is list containing the margin of victory for each game
    schedule: dict, key is team initials, value is dictionary where key is opponent initials, and value is # of games played vs. opponent during period in question

    returns strength of schedule for given team over time period encompassed by margins and schedule.
    '''
    opp_movs = []
    for abbrev, gp in schedule[abbrev].items():
        opp_movs += [margin_for_team(abbrev, margins)] * gp
    return sum(opp_movs) / len(opp_movs)

def weighted_sos_for_team(abbrev, schedule, margins):
    '''
    see sos_for_team, but calculates average margin for each team using weighted_margin_for_team rather than margin_for_team

    abbrev: string, 3 letter initial of NBA team
    margins: dict, key is team initials and value is list containing the margin of victory for each game where first element equals last season average margin of victory
    schedule: dict, key is team initials, value is dictionary where key is opponent initials, and value is # of games played vs. opponent during period in question

    returns weighted strength of schedule for given team over time period encompassed by margins and schedule.
    '''
    opp_movs = []
    for abbrev, gp in schedule[abbrev].items():
        opp_movs += [weighted_margin_for_team(abbrev, margins)] * gp
    return sum(opp_movs) / len(opp_movs)

def average_net_rating_for_team(abbrev, ratings):
    '''
    find the average net rating for a team during a period of time

    abbrev: string, 3 letter initial of NBA team
    ratings: dict, key is team initials and value is list containing net rating for each game played during period in question

    returns the average net rating of given team over time period encompassed by net_ratings
    '''
    return sum(ratings[abbrev]) / len(ratings[abbrev])

def weighted_average_net_rating_for_team(abbrev, ratings):
    '''
    see average_net_rating_for_team, but instead weighs previous season rating by ((82 - games played) / 82) and current season average net rating by (games played / 82)

    abbrev: string, 3 letter initial of NBA team
    ratings: dict, key is team initials and value is list containing net rating for each game played during period in question where first element equals last season's average net rating

    returns the average net rating of given team over time period encompassed by net_ratings
    '''
    last_season_rating = ratings[abbrev][0]
    this_season_ratings = ratings[abbrev][1:]
    gp = len(this_season_ratings)
    if gp == 0:
        # no games played, return last season's average net rating
        return last_season_rating
    elif gp > 82:
        # team in playoffs, use only this season's net ratings
        return sum(this_season_ratings) / len(this_season_ratings)
    else:
        current_season_rating = sum(this_season_ratings) / len(this_season_ratings)
        return (gp / 82 * current_season_rating) + (((82 - gp) / 82)  * last_season_rating)

def is_first_game_of_season(game, abbrev, boxscores_df):
    '''
    determine whether given game is the first game of the season for the given team

    game: pd.Series, see nba_boxscores_1984_2018.csv for format (game is single row)
    abbrev: string, 3 letter initial of NBA team
    boxscores_df: pd.DataFrame, see nba_boxscores_1984_2018.csv for format

    returns boolean
    '''
    first_game_indx = boxscores_df[boxscore_team_mask(boxscores_df, abbrev) & boxscore_season_range_mask(boxscores_df, game["season"], game["season"])].head(1).index[0]
    return game.name == first_game_indx

def regular_season_metrics(abbrev, season, season_summaries_df, rating_cols):
    '''
    determine end of regular season metrics for the given team and given season

    abbrev: string, 3 letter initial of NBA team
    season: int, season to get metrics for (2017 = 2016-17)
    season_summaries_df: pd.DataFrame, see nba_season_summaries_1984_2018.csv for format
    rating_cols: list of metrics desired, see nba_season_summaries_1984_2018.csv columns for possible values

    returns pd.Series with requested metrics
    '''
    team_name = "Charlotte Hornets" if abbrev == "CHO" and season > 2014 else nba_1984_2018_initials_reversed[abbrev]
    return summary_season_query(season_summaries_df, season, team_name, rating_cols)

def abbrev_dict_for_season(season, season_summaries_df):
    '''
    generate a lookup table to map boxscore teams (3 letter initials) to season summary teams (full names)

    season: int, (2017 = 2016-17 NBA season)
    season_summaries_df: pd.DataFrame, see nba_season_summaries_1984_2018.csv for format
    '''
    team_names = season_summaries_df.loc[season].index.tolist()
    team_names.remove("League Average")
    d = {k:v for k, v in nba_1984_2018_initials.items() if k in team_names}
     # handle edge case of boxscores using "CHO" for both Charlotte Bobcats and post 2014 Charlotte Hornets
    if season > 2014:
        d["Charlotte Hornets"] = "CHO"
    return d

def get_season_data(end_year):
    '''
    get cumulative statistics for season specified by end_year

    end_year: int, year to query (ex: 2018 queries 2017-2018 season)

    returns Pandas dataframe w/ basketball-reference.com's miscellaneous stats table for season specified by end_year
    '''
    from bs4 import Comment
    html = "https://www.basketball-reference.com/leagues/NBA_{}.html".format(end_year)
    result = requests.get(html)
    soup = BeautifulSoup(result.content, "html.parser")
    # html tree is strange...table is wrapped inside a comment
    table = [c for c in (soup.find('div', id="all_misc_stats")).children if type(c) == Comment][0]
    # parse table with pandas
    df = pd.read_html(table, header=1)[0]
    df["Season"] = end_year
    return df

def generate_season_summaries_for(start_year, end_year):
    '''
    generate dataframe with basketball-reference season summary data (see https://www.basketball-reference.com/leagues/NBA_2018.html#misc_stats::none) for all seasons [start_year, end_year] (inclusive)

    start_year: int, lower bound for year (ex: 2000 = 1999-2000 NBA season)
    end_year: int, upper bound for year

    returns dataframe with season summary data for all seasons [start_year, end_year] (inclusive)
    '''
    assert start_year >= 1950, "Start year must be 1950 or later"
    df_season_summaries = pd.concat([get_season_data(i) for i in range(start_year, end_year+1)])
    # Playoff teams labeled with * in basketball-reference data. Remove distinction for easier grouping.
    df_season_summaries["Team"] =  df_season_summaries["Team"].map(lambda s: s.replace("*", ""))
    df_season_summaries["NetRtg"] = df_season_summaries["ORtg"] - df_season_summaries["DRtg"]
    df_season_summaries = df_season_summaries.set_index(["Season", "Team"])
    # Re-order columns
    cols = df_season_summaries.columns.values.tolist()
    df_season_summaries = df_season_summaries.reindex_axis(cols[:cols.index("Pace")] + ["NetRtg"] + cols[cols.index("Pace"):-1], axis=1)
    return df_season_summaries

def compute_features_for_game(game_indx, df_boxscores, df_season_summaries, debug=False, weighted=True):
    '''
    generate dataframe with desired model features for all the games in a given season

    game_indx: int, index of game in df_boxscores
    df_boxscores: pd.DataFrame, see nba_boxscores_1984_2018.csv for format
    df_season_summaries: pd.DataFrame, see nba_season_summaries_1984_2018.csv for format
    debug (optional, default=False): boolean, toggle debug print statements
    weighted (optional, default=True): boolean, determines whether SRS and average net rating calculations are weighted according to games played. When true, the average net rating of a team before a game equals (gp/82 * current_season_average_net) + ((82 - gp)/82) * last_season_average_net

    returns model features for game (see Features for model --> Desired for details)
    '''
    season = df_boxscores.loc[game_indx]["season"]
    previous_season = season - 1
    if debug:
        print("Computing features for game: {} vs. {} on {}".format(df_boxscores.loc[game_indx]["team1"], df_boxscores.loc[game_indx]["team2"], df_boxscores.loc[game_indx]["date"].strftime("%Y-%m-%d")))
        print("Performing setup for season")
        print("****************************")
    # Get end of regular season SRS, NetRtg, and MOV for previous season
    last_season_team_names = df_season_summaries.loc[previous_season].index.tolist()
    last_season_team_names.remove("League Average")
    last_season_abbrev_dict = {v:k for k, v in abbrev_dict_for_season(previous_season, df_season_summaries).items()}
    last_season_metrics = summary_season_query(df_season_summaries, previous_season, last_season_team_names, ["SRS", "NetRtg", "MOV"]).loc[previous_season]
    if debug:
        print("Successfully received metrics for {} season".format(previous_season))
    # dictionary for mapping between team initials (used in boxscore) and full team names (used in summary)
    this_season_abbrev_dict = {v:k for k, v in abbrev_dict_for_season(season, df_season_summaries).items()}
    if debug:
        print("Successfully created abbreviation mapping for {} season".format(season))
    # dictionaries for storing margins of victory and schedule (used for SRS) and net ratings for each team on a per-game basis (used for avg. NetRtg)
    margins = {}
    schedule = {}
    net_ratings = {}
    for abbrev in this_season_abbrev_dict.keys():
        margins[abbrev] = []
        schedule[abbrev] = {}
        net_ratings[abbrev] = []
    season_start_date = df_boxscores.loc[df_boxscores.index[0], "date"]
    end_date = df_boxscores.loc[game_indx]["date"]
    if debug:
        print("Successfully initialized dictionaries for storing MOV, schedule, and net ratings for each team on per-game basis")
        print("Finished setup for season")
        print("****************************")
        print("Beginning walkthrough of games from {} to {}".format(season_start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")))
        i = 0
    # walk through current season game by game, keeping track of win margins, schedules, and net rating
    for _, game in df_boxscores[boxscore_date_range_mask(df_boxscores, season_start_date, end_date)].iterrows():
        team1 = game["team1"]
        team1_score = game["score1"]
        team2 = game["team2"]
        team2_score = game["score2"]
        if debug:
            print("Game {}".format(i))
            print("****************************")
            print("{} at {}: {} - {}".format(team2, team1, team2_score, team1_score))
        # determine pre-game SRS and NetRtg for home and away team
        for t, key_prefix in zip([team1, team2], ["team1", "team2"]):
            if is_first_game_of_season(game, t, df_boxscores):
                if debug:
                    print("First game of season for {}. Attempt to use last season's metrics".format(t))
                # first game of season for team, use last season's SRS and NetRtg values. Add NetRtg and MOV from last season as first element of current season tally to reduce variance for early season games.
                try:
                    net = last_season_metrics.loc[last_season_abbrev_dict[t]]["NetRtg"]
                    mov = last_season_metrics.loc[last_season_abbrev_dict[t]]["MOV"]
                    net_ratings[t].append(net)
                    margins[t].append(mov)
                    if debug:
                        print("Added NetRtg: {} and MOV: {} to per-game dictionaries for {}".format(net, mov, t))
                except KeyError:
                    if debug:
                        print("Expansion team, no results avaiable from last season. Attempt to use lower quartile results of previous season")
                    # first season of expansion franchise. Set to lower quartile value of previous season
                    net = last_season_metrics["NetRtg"].quantile(0.25)
                    mov = last_season_metrics["MOV"].quantile(0.25)
                    net_ratings[t].append(net)
                    margins[t].append(mov)
                    if debug:
                        print("Added NetRtg: {} and MOV: {} to per-game dictionaries for {}".format(net, mov, t))

        if game.name == game_indx:
            if debug:
                print("Found target game: {} vs. {} on {}".format(team1, team2, game["date"].strftime("%Y-%m-%d")))
            elo_diff = game["elo1_pre"] - game["elo2_pre"]
            net_team1 = weighted_average_net_rating_for_team(team1, net_ratings) if weighted else average_net_rating_for_team(team1, net_ratings)
            net_team2 = weighted_average_net_rating_for_team(team2, net_ratings) if weighted else average_net_rating_for_team(team2, net_ratings)
            net_diff = net_team1 - net_team2
            try:
                srs_team1 = (weighted_margin_for_team(team1, margins) + weighted_sos_for_team(team1, schedule, margins)) if weighted else (margin_for_team(team1, margins) + sos_for_team(team1, schedule, margins))
                srs_team2 = (weighted_margin_for_team(team2, margins) + weighted_sos_for_team(team2, schedule, margins)) if weighted else (margin_for_team(team2, margins) + sos_for_team(team2, schedule, margins))
            except:
                # first game of season, use last season's SRS
                srs_team1 = last_season_metrics.loc[last_season_abbrev_dict[team1]]["SRS"]
                srs_team2 = last_season_metrics.loc[last_season_abbrev_dict[team2]]["SRS"]
            srs_diff = srs_team1 - srs_team2
            if debug:
                    print("ELO diff = {}".format(elo_diff))
                    print("NetRtg diff = {}".format(net_diff))
                    print("SRS diff = {}".format(srs_diff))
            return [elo_diff, net_diff, srs_diff]

        # update margins
        mov_team1 = team1_score - team2_score
        mov_team2 = -mov_team1
        margins[team1].append(mov_team1)
        margins[team2].append(mov_team2)
        if debug:
            print("Updated margins for {}: {}, {}: {}".format(team1, mov_team1, team2, mov_team2))
        # update schedule
        gp = schedule[team1].get(team2, None)
        if gp is None:
            schedule[team1][team2] = 1
            schedule[team2][team1] = 1
        else:
            schedule[team1][team2] = gp + 1
            schedule[team2][team1] = gp + 1
        if debug:
            print("Updated schedule. {} now played {} {} times, and {} played {} {} times".format(team1, team2, schedule[team1][team2], team2, team1, schedule[team2][team1]))
        # update net ratings
        net_team1 = game["team1_NetRtg"]
        net_team2 = game["team2_NetRtg"]
        net_ratings[team1].append(net_team1)
        net_ratings[team2].append(net_team2)
        if debug:
            print("Updated net ratings for {}: {}, {}: {}".format(team1, net_team1, team2, net_team2))
            i += 1
            print("****************************")

def predict_win_probability_for_game(home_team_initials, away_team_initials, date, boxscores_csv_path=None, season_summaries_csv_path=None):
    '''
    Use weighted logistic regression to predict the probability the home team wins the given game

    home_team_initials: string, 3 letter abbreviation for home team (basketball-reference boxscore initials)
    away_team_initials: string, 3 letter abbreviation for away team (basketball-reference boxscore initials)
    date: string, the date of the game to predict (format=YYYY-MM-DD)
    boxscores_csv_path: string, path to boxscores csv, see nba_boxscores_1984_2018.csv for format
    season_summaries_csv_path: string, path to season summaries csv, see nba_season_summaries_1984_2018.csv for format

    ex: predict_win_probability_for_game("NYK", "DAL", 2018-03-14)
    '''
    # parse date argument
    try:
        date = datetime.datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise Exception("Date {} does not match format YYYY-MM-DD".format(date))
    season = date.year if date.month in list(range(1,7)) else date.year + 1

    # check if boxscores dataset is present
    if boxscores_csv_path:
        # check given dataset for game in question
        df_boxscores = pd.read_csv(boxscores_csv_path, index_col=0, parse_dates=[1], infer_datetime_format=True)
        df_boxscores = df_boxscores[boxscore_season_range_mask(df_boxscores, season, season)]
        df_boxscores = df_boxscores.reset_index(drop=True)
        try:
            game_indx = df_boxscores[(df_boxscores["date"] == date) & (df_boxscores["team1"] == home_team_initials) & (df_boxscores["team2"] == away_team_initials)].index[0]
        except IndexError:
            raise Exception("No game found between {} and {} on {} in \"{}\" dataset".format(home_team_initials, away_team_initials, date.strftime("%Y-%m-%d"), boxscores_csv_path))
    else:
        # download boxscores for same season as game in question
        print("DOWNLOADING BOXSCORES")
        if not os.path.isdir("../Data"):
            os.mkdir("../Data")
        session = subprocess.Popen(['python', 'scrape_boxscores.py', season, season], stdout=PIPE, stderr=PIPE)
        stdout, stderr = session.communicate()
        if stderr:
            raise Exception("Error "+str(stderr))
        # if execution reaches here, the data was downloaded to ../Data/nba_game_data_YYYY-MM-DD_to_YYYY-MM-DD.csv, where first date is date of first game of season and last date is date of final game of season
        fn = glob.glob("../Data/nba_game_data_{}-*-*_to_{}-*-*.csv".format(season-1, season))
        if not fn:
            raise Exception("Could not find file matching ../Data/nba_game_data_{}-*-*_to_{}-*-*.csv".format(season-1, season))
        df_boxscores = pd.read_csv(fn, index_col=0, parse_dates=[1], infer_datetime_format=True)
        df_boxscores = df_boxscores.dropna(subset=["score1", "score2", "team1_NetRtg", "team2_NetRtg", "elo1_pre", "elo2_pre"])

    # check if season summaries dataset is present
    if season_summaries_csv_path:
        df_season_summaries = pd.read_csv(season_summaries_csv_path, index_col=[0, 1])
        df_season_summaries.sort_index(inplace=True)
        try:
            season_indx = df_season_summaries.loc[season-1]
        except:
            raise Exception("No season summary found for {} season in \"{}\" dataset".format(season-1, season_summaries_csv_path))
    else:
        # download season summary for previous season
        print("DOWNLOADING SEASON SUMMARIES")
        df_season_summaries = generate_season_summaries_for(season-1, season-1)
        df_season_summaries.sort_index(inplace=True)

    # compute features for game
    x = compute_features_for_game(game_indx, df_boxscores, df_season_summaries, debug=False, weighted=True)
    clf = joblib.load("../Data/best_weighted_clf_all_data.pkl")
    return clf.predict_proba([x])[0][1]

if __name__ == "__main__":
    try:
        home_team = sys.argv[1]
        away_team = sys.argv[2]
        date_str = sys.argv[3]
    except:
        raise Exception(
        '''
        arguments:
          home_team - 3 letter abbreviation for home team (basketball-reference boxscore initials)
          away_team - 3 letter abbreviation for away team (basketball-reference boxscore initials)
          game date - the date of the game to predict (format=YYYY-MM-DD)
          (optional) boxscores_csv_path: string, path to boxscores csv, see nba_boxscores_1984_2018.csv for format
          (optional) season_summaries_csv_path: string, path to season summaries csv, see nba_season_summaries_1984_2018.csv for format

          WARNING: if boxscores_csv_path or season_summaries_csv_path are not provided, the script will attempt to scrape the needed data. This requires an internet connection and will take time. THIS FEATURE IS CURRENTLY UNTESTED AND MAY NOT FUNCTION PROPERLY!

        ex: python single_game_prediction.py MIA CHO 2016-10-28 ../Data/nba_boxscores_1984_2018.csv ../Data/nba_season_summaries_1984_2018.csv
        ''')
    try:
        boxscores_csv_path = sys.argv[4]
        season_summaries_csv_path = sys.argv[5]
    except:
        print("No paths found for one or both of boxscores and season summaries datasets. WARNING: Attempting to scrape the needed data. This requires an internet connection and will take time. THIS FEATURE IS CURRENTLY UNTESTED AND MAY NOT FUNCTION PROPERLY!")

    print("{} win probability: {}".format(home_team, predict_win_probability_for_game(home_team, away_team, date_str, boxscores_csv_path=boxscores_csv_path, season_summaries_csv_path=season_summaries_csv_path)))
