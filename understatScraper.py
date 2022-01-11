

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv) 
import pandas_gbq as pd_gbq
import requests 
from bs4 import BeautifulSoup
import json 
from random import random
from time import sleep
import datetime as d


# create urls for all seasons of all leagues 
base_url = 'https://understat.com' 
leagues = ['EPL', 'La_liga', 'Bundesliga', 'Serie_A', 'Ligue_1', 'RFPL'] 
seasons = ['2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']


def scrape_match_ids():
    url = base_url+'/'+'sitemap.xml'
    res = requests.get(url) 
    soup = BeautifulSoup(res.content, "lxml")
    sites = soup.text.split(sep = 'https://')
    match_ids = []
    for site in sites:
        if 'understat.com/match/' in site:
            match_ids.append(site[20:])
    return match_ids


def scrape_leaguetable(league, year):

    url = base_url+'/'+'league/'+league+'/'+year
    res = requests.get(url) 
    soup = BeautifulSoup(res.content, "lxml")
    # Based on the structure of the webpage, I found that data is in the JSON variable, under 'script' tags 
    scripts = soup.find_all('script')


    # Extract the JSON string from the script tags
    string_with_json_obj = '' 
    # Find data for teams 
    for el in scripts: 
        if len(el.contents) != 0:
            if 'teamsData' in el.contents[0]: 
                string_with_json_obj = el.contents[0].strip()
        
        
    #   strip unnecessary symbols and get only JSON data 
    ind_start = string_with_json_obj.index("('")+2 
    ind_end = string_with_json_obj.index("')") 
    json_data = string_with_json_obj[ind_start:ind_end] 
    json_data = json_data.encode('utf8').decode('unicode_escape')

    # convert json data to initial pandas dataframe
    a_json = pd.DataFrame.from_dict(json.loads(json_data), orient = 'index').reset_index().drop(columns = ['index'])

    # expand the history field (currently a list of dictionaries) in multiple rows and append with team ID and name
    team_data = pd.DataFrame()
    
    for index, row in a_json.iterrows():
            team_row = pd.DataFrame.from_dict(row['history'])
            team_row['team'] = row['title']
            team_row['id'] = row['id']
            team_data = team_data.append(team_row)
                
    # expand ppda & ppda columns (dictionaries) into seperate columns with ppda prefix
    
    ppda = pd.json_normalize(team_data['ppda']).add_prefix('ppda_')
    ppda_allowed = pd.json_normalize(team_data['ppda_allowed']).add_prefix('ppda_allowed_')

    team_data = team_data.drop(columns=['ppda','ppda_allowed']).join([ppda, ppda_allowed ])

    return team_data


def scrape_league_table_history():
    leaguetable_history_df = pd.DataFrame()

    for league in leagues:
        for season in seasons:
            leaguetable_df = scrape_leaguetable(league, season)
            leaguetable_df['year'] = season
            leaguetable_df['league'] = league
            leaguetable_history_df =  leaguetable_history_df.append(leaguetable_df)
            sleep(random())
    return leaguetable_history_df




def scrape_players_data(league, year):
    url = base_url+'/'+'league/'+league+'/'+year
    res = requests.get(url) 
    soup = BeautifulSoup(res.content, "lxml")
    # Based on the structure of the webpage, I found that data is in the JSON variable, under 'script' tags 
    scripts = soup.find_all('script')


    # Extract the JSON string from the script tags
    string_with_json_obj = '' 
    # Find data for teams 
    for el in scripts: 
        if len(el.contents) != 0:
            if 'playersData' in el.contents[0]: 
                string_with_json_obj = el.contents[0].strip()
       
        
    #   strip unnecessary symbols and get only JSON data 
    ind_start = string_with_json_obj.index("('")+2 
    ind_end = string_with_json_obj.index("')") 
    json_data = string_with_json_obj[ind_start:ind_end] 
    player_data = pd.DataFrame(eval(json_data.encode('utf8').decode('unicode_escape')))
    return player_data    
       
def scrape_players_history():
    playerhistory_df = pd.DataFrame()
    for league in leagues:
        for season in seasons:
            player_df = scrape_players_data(league, season)
            player_df['year'] = season
            player_df['league'] = league
            playerhistory_df = playerhistory_df.append(player_df)
            sleep(random())
    return playerhistory_df





    
def scrape_match(match_id):
    url = base_url+'/'+'match'+'/'+str(match_id)
    res = requests.get(url) 
    soup = BeautifulSoup(res.content, "lxml")
    # Based on the structure of the webpage, I found that data is in the JSON variable, under 'script' tags 
    scripts = soup.find_all('script')


    # Extract the JSON string from the script tags
    string_with_json_obj = '' 
    # Find data for teams 
    for el in scripts: 
        if len(el.contents) != 0:
            if 'rostersData' in el.contents[0]: 
                string_with_json_obj = el.contents[0].strip()

    ind_start = string_with_json_obj.index("('")+2 
    ind_end = string_with_json_obj.index("')") 
    json_data = string_with_json_obj[ind_start:ind_end] 
    player_data = eval(json_data.encode('utf8').decode('unicode_escape'))

    labels = soup.find_all('label')

    home = ''
    away = ''

    for label in labels:
        if label.attrs['for'] == 'team-away':
            away = label.text
        elif label.attrs['for'] == 'team-home':
            home = label.text
    
    away_data = pd.DataFrame.from_dict(player_data['a'], orient = 'index')
    away_data['team'] = away

    home_data = pd.DataFrame.from_dict(player_data['h'], orient = 'index')
    home_data['team'] = home
    
    game_data =  pd.concat([away_data, home_data])
    
    
    date_text = soup.find('title').text
    date_string = date_text[date_text.find('(')+1:date_text.find(')')]
    date = d.datetime.strptime(date_string, '%B %d %Y') 
    
    game_data['match_date'] = date
    game_data['match_id'] = match_id
    return game_data
    
    
def max_val(csv, col):
    df = pd.read_csv(csv)    
    return df[col].max()
    
    
def batch_scrape_matches(match_ids):
    df = pd.DataFrame()
    for batch_id, match_id in enumerate(match_ids):
        try:
            df = df.append(scrape_match(match_id))
        except:
            continue
        finally:
            if batch_id % 20 == 0:
                print('Completed ' + str(batch_id) + ' sites')
            sleep(random())
            batch_id += 1
    return df


def update_match_data():
    df = batch_scrape_matches('match_data.csv')
    df.to_csv('match_data.csv', mode = 'a', header=False, index = False)



# Global project ID for all football stats
project_id = 'footballstats-337117'

# Generate the league table data from the scrapers
table_data = scrape_league_table_history()

# League table name in GCP
leaguetable_table_id = 'understat_data.tb_table_history'

# Re-write the league table data to GCP
pd_gbq.to_gbq(table_data, leaguetable_table_id, project_id = project_id, if_exists='replace')


# Generate the players data from the scrapers
player_data = scrape_players_history()

# Players table name in GCP
player_table_id = 'understat_data.tb_player_history'

# Re-Wrte the player table data to GCP
pd_gbq.to_gbq(player_data, player_table_id, project_id = project_id, if_exists='replace')


# Read SQL query to pull the currently stored player data
sql_query = 'SELECT DISTINCT match_id FROM `footballstats-337117.understat_data.tb_match_history`'

# Read from GBQ using the pandas integrated functions
stored_match_id = pd_gbq.read_gbq(sql_query, project_id = project_id)['match_id'].to_list()

# Scrape the sitemap to understand what match IDs we might be missing 
live_match_id = scrape_match_ids()

match_ids = list(set(live_match_id) - set(stored_match_id))

new_match_data = batch_scrape_matches(match_ids)







match_table_id = 'understat_data.tb_match_history'
    
if len(new_match_data) != 0:
    new_match_data['match_date'] = new_match_data['match_date'].astype('str')
    new_match_data['match_id'] = new_match_data['match_id'].astype('str')
    pd_gbq.to_gbq(new_match_data, match_table_id, project_id = project_id, if_exists='append')




