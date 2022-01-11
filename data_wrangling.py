
import pandas as pd
import pandas_gbq as pd_gbq
import numpy as np

project_id = 'footballstats-337117'

## First we want to get a list of all the fixtures, on which we can add out features and predictors


## SQL function to pass to Google Big Query

fixture_list_sql = """
SELECT  DISTINCT 
        a.*
    ,   b.Team
FROM (  SELECT DISTINCT Match_ID, Match_date, MIN(Team) AS team
        FROM `footballstats-337117.understat_data.tb_match_history`
        WHERE Team IN ( SELECT DISTINCT Team 
                FROM `footballstats-337117.understat_data.tb_table_history`
                WHERE league = 'EPL')
        GROUP BY match_date, match_id
) a
LEFT JOIN `footballstats-337117.understat_data.tb_match_history` b ON a.Match_ID = b.match_id
                                                                AND a.Team != b.Team
"""

# We want 2 rows per game, one for each goal-scored regression

# We can get this by creating a copy of the match, swap the team columns and then concatenate back on

fixtures = pd_gbq.read_gbq(fixture_list_sql, project_id = project_id)

df = fixtures[['Match_ID','Match_date','Team_1','team']]

df = df.rename(columns = {'Match_ID':'Match_ID','Match_date':'Match_date','Team_1':'team','team':'Team_1'})

fixtures = fixtures.append(df)

fixtures = fixtures.sort_values('Match_ID')




# Now we want to pull and aggregate the match history data to understand historic team performances


match_data_sql = """
SELECT  a.*
FROM    `footballstats-337117.understat_data.tb_match_history` a 
INNER JOIN (SELECT DISTINCT Team 
            FROM `footballstats-337117.understat_data.tb_table_history`
            WHERE league = 'EPL') b ON a.Team = b.Team
"""

match_data = pd_gbq.read_gbq(match_data_sql, project_id = project_id)


dtypes = {'goals':'float64','shots':'float64','xG':'float64','key_passes':'float64','assists':'float64','xA':'float64','xGChain':'float64','xGBuildup':'float64'}
match_data = match_data.astype(dtypes)


team_match_data = match_data.groupby(['match_date', 'team','match_id','h_a']).agg({'goals':'sum', 'shots':'sum', 'xG':'sum','key_passes':'sum', 'assists':'sum', 'xA':'sum','xGChain':'sum','xGBuildup':'sum'}).reset_index()


team_match_data['match_rank'] = team_match_data.groupby('team')['match_date'].rank(method = 'first', ascending = False)

## Let's add this team game index onto the fixtures data so that we can pull data between the tables

fixtures = fixtures.merge(team_match_data[['team','match_id','match_rank']], how = 'left', left_on = ['team','Match_ID'], right_on = ['team', 'match_id'])

fixtures.drop(columns = ['match_id'],inplace = True )


# Finally we want to pull the table history data


table_data_sql = """
                SELECT *
                FROM `footballstats-337117.understat_data.tb_table_history`
                WHERE league = 'EPL'
"""

table_data = pd_gbq.read_gbq(table_data_sql, project_id = project_id)

table_data['date'] = pd.to_datetime(table_data['date']).dt.date

fixtures['short_date'] = fixtures['Match_date'].dt.date

table_data = table_data.merge(fixtures[['team','short_date','match_rank']], how = 'left',left_on = ['team','date'], right_on = ['team','short_date'])



def historic_team_stat(df, feature_table, feature, lag, opponent = False):
    mod_data = feature_table.copy()
    mod_data['match_rank'] = mod_data['match_rank'] - lag
    if opponent:
        join_df = df.merge(mod_data, how = 'left', left_on = ['Team_1','match_rank'], right_on = ['team','match_rank'])
        return join_df[feature]
    else:
        join_df = df.merge(mod_data, how = 'left', left_on = ['team','match_rank'], right_on = ['team','match_rank'])
        return join_df[feature]



fixtures['h_a'] = historic_team_stat(fixtures,team_match_data, 'h_a', 0)

fixtures['xG lag 1'] = historic_team_stat(fixtures,team_match_data, 'xG', 1)
fixtures['xG lag 2'] = historic_team_stat(fixtures,team_match_data, 'xG', 2)
fixtures['xG lag 3'] = historic_team_stat(fixtures,team_match_data, 'xG', 3)

fixtures['goals lag 1'] = historic_team_stat(fixtures, team_match_data,'goals', 1)
fixtures['goals lag 2'] = historic_team_stat(fixtures, team_match_data,'goals', 2)
fixtures['goals lag 3'] = historic_team_stat(fixtures, team_match_data,'goals', 3)

fixtures['wins lag 1'] = historic_team_stat(fixtures, table_data,'wins', 1)
fixtures['wins lag 2'] = historic_team_stat(fixtures, table_data,'wins', 2)
fixtures['wins lag 3'] = historic_team_stat(fixtures, table_data,'wins', 3)
fixtures['wins lag 4'] = historic_team_stat(fixtures, table_data,'wins', 4)
fixtures['wins lag 5'] = historic_team_stat(fixtures, table_data,'wins', 5)

fixtures['win_rate'] = (fixtures['wins lag 1'].replace(np.nan, 0) + fixtures['wins lag 2'].replace(np.nan, 0) + fixtures['wins lag 3'].replace(np.nan, 0) + fixtures['wins lag 4'].replace(np.nan, 0) + fixtures['wins lag 5'].replace(np.nan, 0)) / 5.0

fixtures = fixtures.drop(columns = ['wins lag 1','wins lag 2','wins lag 3','wins lag 4','wins lag 5'])

fixtures['draws lag 1'] = historic_team_stat(fixtures, table_data,'draws', 1)
fixtures['draws lag 2'] = historic_team_stat(fixtures, table_data,'draws', 2)
fixtures['draws lag 3'] = historic_team_stat(fixtures, table_data,'draws', 3)
fixtures['draws lag 4'] = historic_team_stat(fixtures, table_data,'draws', 4)
fixtures['draws lag 5'] = historic_team_stat(fixtures, table_data,'draws', 5)

fixtures['draw_rate'] = (fixtures['draws lag 1'].replace(np.nan, 0) + fixtures['draws lag 2'].replace(np.nan, 0) + fixtures['draws lag 3'].replace(np.nan, 0) + fixtures['draws lag 4'].replace(np.nan, 0) + fixtures['draws lag 5'].replace(np.nan, 0)) / 5.0

fixtures = fixtures.drop(columns = ['draws lag 1','draws lag 2','draws lag 3','draws lag 4','draws lag 5'])


fixtures['loses lag 1'] = historic_team_stat(fixtures, table_data,'loses', 1)
fixtures['loses lag 2'] = historic_team_stat(fixtures, table_data,'loses', 2)
fixtures['loses lag 3'] = historic_team_stat(fixtures, table_data,'loses', 3)
fixtures['loses lag 4'] = historic_team_stat(fixtures, table_data,'loses', 4)
fixtures['loses lag 5'] = historic_team_stat(fixtures, table_data,'loses', 5)

fixtures['loss_rate'] = (fixtures['loses lag 1'].replace(np.nan, 0) + fixtures['loses lag 2'].replace(np.nan, 0) + fixtures['loses lag 3'].replace(np.nan, 0) + fixtures['loses lag 4'].replace(np.nan, 0) + fixtures['loses lag 5'].replace(np.nan, 0)) / 5.0

fixtures = fixtures.drop(columns = ['loses lag 1','loses lag 2','loses lag 3','loses lag 4','loses lag 5'])

fixtures['ppda_att'] = historic_team_stat(fixtures, table_data,'ppda_att', 1)
fixtures['ppda_def'] = historic_team_stat(fixtures, table_data,'ppda_def', 1)
fixtures['ppda_allowed_def'] = historic_team_stat(fixtures, table_data,'ppda_allowed_def', 1)
fixtures['ppda_allowed_att'] = historic_team_stat(fixtures, table_data,'ppda_allowed_att', 1)



fixtures['ppda_att_opp'] = historic_team_stat(fixtures, table_data,'ppda_att_y', 1,opponent = True)
fixtures['ppda_def_opp'] = historic_team_stat(fixtures, table_data,'ppda_def_y', 1,opponent = True)
fixtures['ppda_allowed_def_opp'] = historic_team_stat(fixtures, table_data,'ppda_allowed_def_y', 1,opponent = True)
fixtures['ppda_allowed_att_opp'] = historic_team_stat(fixtures, table_data,'ppda_allowed_att_y', 1,opponent = True)



fixtures['goals'] = historic_team_stat(fixtures, team_match_data,'goals', 0)


fixtures.to_csv('training_data.csv', index = False)

