import sys
import datetime
import time

import pandas as pd
from bs4 import BeautifulSoup
import requests

def boxscore_links_for_date(date):
    '''
    get list of basketball-reference links to boxscores for games on given date

    date: datetime.datetime object with year, month, and day specified

    returns list of urls to basketball-reference single game boxscores for given date
    '''
    link = "https://www.basketball-reference.com/boxscores/?month={}&day={}&year={}".format(date.month, date.day, date.year)
    result = requests.get(link)
    soup = BeautifulSoup(result.content, "html.parser")
    return ["http://www.basketball-reference.com" + game.find("a").get("href") for game in soup.find_all("td", {"class": "right gamelink"})]

def boxscore_dict_for_link(link):
    '''
    advanced box score stats as dictionary from basketball-reference boxscore link

    link: string, link to single game basketball-reference boxscore

    returns dictionary with advanced stats for home (team1) and away (team2) teams for boxscore linked
    NOTE: basketball-reference only supports single game advanced stats starting from 1983-1984 NBA season
    '''
    result = requests.get(link)
    soup = BeautifulSoup(result.content, "html.parser")
    from bs4 import Comment
    assert soup.find("div", id="all_four_factors") != None, "Advanced box score metrics only available for dates with at least 1 NBA game starting from 1983-1984 season"
    for c in soup.find("div", id="all_four_factors").children:
        if type(c) == Comment:
            s_ind = c.index("<table")
            e_ind = c.index("</table>")
            table_html = c[s_ind:e_ind+8]
            break
    df = pd.read_html(table_html, header=1, index_col=0)[0]
    df["DRtg"] = df["ORtg"].values[::-1]
    df["NetRtg"] = df["ORtg"] - df["DRtg"]
    d = {}
    for i, team_name in enumerate(df.index):
        prefix = "team2_" if i == 0 else "team1_"
        for col in df.columns:
            d[prefix+col] = df.loc[team_name, col]
    return d

if __name__ == "__main__":
    # Basketball-reference.com has advanced metrics for single games starting w/ 1983-1984 NBA season
    start_year = int(sys.argv[1])
    end_year = int(sys.argv[2])
    print(start_year, end_year)
    url = "https://projects.fivethirtyeight.com/nba-model/nba_elo.csv"
    df = pd.read_csv(url).astype({'date': 'datetime64[ns]'})
    season_start_dates = []
    season_end_dates = []
    for i in range(start_year, end_year+1):
        season_df = df[(df["season"] == i) & (~pd.isnull(df["score1"]))]
        season_start_dates.append(season_df.loc[season_df.index[0], "date"])
        season_end_dates.append(season_df.loc[season_df.index[-1], "date"])
    # Query basketball reference game by game, and save results for each year
    for season_start_date, season_end_date in zip(season_start_dates, season_end_dates):
        df_slice = df[(df["date"] >= season_start_date) & (df["date"] <= season_end_date)].copy()
        current_date = season_start_date
        boxscores_for_date = []
        while not boxscores_for_date:
            try:
                boxscores_for_date = boxscore_links_for_date(current_date)
            except:
                print(sys.exc_info())
                print("Error occured. Sleeping for 1 min and re-trying")
                boxscores_for_date = []
                time.sleep(60)
        data = []
        for _, row in df_slice.iterrows():
            print("{} vs. {} on {}".format(row["team1"], row["team2"], row["date"]))
            if current_date != row["date"]:
                # get new boxscores for date
                print("New day ({}), getting boxscores".format(row["date"]))
                boxscores_for_date = []
                while not boxscores_for_date:
                    try:
                        boxscores_for_date = boxscore_links_for_date(row["date"])
                    except:
                        print(sys.exc_info())
                        print("Error occured. Sleeping for 1 min and re-trying")
                        boxscores_for_date = []
                        time.sleep(60)
                current_date = row["date"]
            team1 = row["team1"]
            team2 = row["team2"]
            if team1 == "NOP" and season_start_date.year >= 2002 and season_end_date.year <= 2013:
                team1 = "NOH"
            elif team2 == "NOP" and season_start_date.year >= 2002 and season_end_date.year <= 2013:
                team2 = "NOH"
            filtered_boxscores = list(filter(lambda link: team1 in link, boxscores_for_date))
            if not filtered_boxscores:
                filtered_boxscores = list(filter(lambda link: team2 in link, boxscores_for_date))
            else:
                boxscore_for_game = filtered_boxscores[0]
            d = {}
            while not d:
                try:
                    d = boxscore_dict_for_link(boxscore_for_game)
                except:
                    print(sys.exc_info())
                    print("Error occured. Sleeping for 1 min and re-trying")
                    d = {}
                    time.sleep(60)
            for key, val in d.items():
                row[key] = val
            data.append(row)
        # save data for year
        print("Saving for season from {} to {}".format(season_start_date, season_end_date))
        pd.DataFrame(data).to_csv("../Data/nba_game_data_{}-{}-{}_to_{}-{}-{}.csv".format(season_start_date.year, season_start_date.month, season_start_date.day, season_end_date.year, season_end_date.month, season_end_date.day))
