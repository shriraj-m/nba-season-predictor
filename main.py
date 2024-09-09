import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score

pd.set_option('display.max_rows', None)

prev_games = pd.read_csv("data/games.csv", index_col=0)
nba24_25_matches = pd.read_csv("data/2024-25-NBA-SEASON.csv", index_col=0)

prev_games["date"] = pd.to_datetime(prev_games["game_date"])
nba24_25_matches["off_date"] = pd.to_datetime(nba24_25_matches["DATE"])

# prev_games["wl_home_code"] = prev_games["wl_home"].astype("category").cat.codes
# prev_games["away_team"] = prev_games["team_abbreviation_away"].astype("category").cat.codes
# prev_games["home_team"] = prev_games["team_abbreviation_home"].astype("category").cat.codes
prev_games["target_home"] = (prev_games["wl_home"] == "W").astype("int")
prev_games["target_away"] = (prev_games["wl_away"] == "W").astype("int")
all_teams = pd.concat([
    prev_games[['team_abbreviation_away', 'team_abbreviation_home']],
    nba24_25_matches[['TEAM 1', 'TEAM 2']]
]).stack().unique()  # Combine all team names

team_mapping = pd.Series(all_teams).astype('category').cat.codes
team_mapping_dict = pd.Series(team_mapping.values, index=all_teams).to_dict()

# Apply the same encoding to both datasets
prev_games['away_team'] = prev_games['team_abbreviation_away'].map(team_mapping_dict)
prev_games['home_team'] = prev_games['team_abbreviation_home'].map(team_mapping_dict)

nba24_25_matches['away_team'] = nba24_25_matches['TEAM 1'].map(team_mapping_dict)
nba24_25_matches['home_team'] = nba24_25_matches['TEAM 2'].map(team_mapping_dict)

# nba24_25_matches["away_team"] = nba24_25_matches["TEAM 1"].astype("category").cat.codes
# nba24_25_matches["home_team"] = nba24_25_matches["TEAM 2"].astype("category").cat.codes


rf_home = RandomForestClassifier(n_estimators=100,
                                 min_samples_split=10,
                                 random_state=1,
                                 max_depth=10)
rf_away = RandomForestClassifier(n_estimators=100,
                                 min_samples_split=10,
                                 random_state=1,
                                 max_depth=10)

train = prev_games[prev_games["date"] > '2016-08-01']
test = nba24_25_matches[nba24_25_matches["off_date"] < '2025-08-01']

predictors = ["away_team", "home_team"]

rf_home.fit(train[predictors], train["target_home"])
rf_away.fit(train[predictors], train["target_away"])

predictions_home = rf_home.predict(test[predictors])
predictions_away = rf_away.predict(test[predictors])


def win_loss_prediction(home_or_away):
    wl_prediction = []
    for x in home_or_away:
        if x == 1:
            wl_prediction.append("W")
        else:
            wl_prediction.append("L")
    return wl_prediction


wl_prediction_home = win_loss_prediction(predictions_home)
wl_prediction_away = win_loss_prediction(predictions_away)

nba24_25_matches["predict_wl_home"] = wl_prediction_home
nba24_25_matches["predict_wl_away"] = wl_prediction_away

display_columns = ["DATE", "TEAM 1", "TEAM 2", "predict_wl_home"]


# print(nba24_25_matches[display_columns])
#
# specific_team = "BOS"
# print(nba24_25_matches[nba24_25_matches["TEAM 2"] == specific_team])
# print(nba24_25_matches[nba24_25_matches["TEAM 1"] == specific_team])


def get_record(team_name):
    team_wins_home = nba24_25_matches[
        (nba24_25_matches["TEAM 2"] == team_name) &
        (nba24_25_matches["predict_wl_home"] == "W")
        ]

    team_losses_home = nba24_25_matches[
        (nba24_25_matches["TEAM 2"] == team_name) &
        (nba24_25_matches["predict_wl_home"] == "L")
        ]

    team_wins_away = nba24_25_matches[
        (nba24_25_matches["TEAM 1"] == team_name) &
        (nba24_25_matches["predict_wl_away"] == "W")
        ]

    team_losses_away = nba24_25_matches[
        (nba24_25_matches["TEAM 1"] == team_name) &
        (nba24_25_matches["predict_wl_away"] == "L")
        ]

    win_count_home = team_wins_home.shape[0]
    win_count_away = team_wins_away.shape[0]
    loss_count_home = team_losses_home.shape[0]
    loss_count_away = team_losses_away.shape[0]

    # print(f"Home: {win_count_home}-{loss_count_home}")
    # print(f"Away: {win_count_away}-{loss_count_away}")

    total_wins = win_count_home + win_count_away
    total_loss = loss_count_home + loss_count_away
    return total_wins, total_loss


print(get_record("DET"))
print(get_record("WAS"))
team_abbreviations = nba24_25_matches["TEAM 1"].unique()
'''
Find Overall Record:
'''
overall_record = {}
for x in team_abbreviations:
    overall_record[x] = get_record(x)

sorted_record = dict(sorted(overall_record.items(), key=lambda overall_record: overall_record[1], reverse=True))

for y in sorted_record:
    print(f"{y.upper()}: {sorted_record[y][0]}-{sorted_record[y][1]}")

