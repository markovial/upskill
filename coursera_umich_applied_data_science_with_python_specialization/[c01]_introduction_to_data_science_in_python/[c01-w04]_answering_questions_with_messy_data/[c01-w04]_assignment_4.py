'''

Correlations between location of NHL , NBA , MLB , NFL team city locations , and thier Win loss ratio

'''

import pandas as pd
import numpy as np
import scipy.stats as stats
import re
import os

# Def : df_read {{{
def df_read( path ):

    dir_name  = os.path.dirname  ( __file__       )
    file_name = os.path.join     ( dir_name, path )
    file_type = os.path.splitext ( file_name ) [1][1:]

    if   ( file_type == "csv"  ) : df = pd.read_csv     ( file_name )
    elif ( file_type == "txt"  ) : df = pd.read_csv     ( file_name )
    elif ( file_type == "zip"  ) : df = pd.read_csv     ( file_name )
    elif ( file_type == "xlsx" ) : df = pd.read_excel   ( file_name )
    elif ( file_type == "json" ) : df = pd.read_json    ( file_name )
    elif ( file_type == "html" ) : df = pd.read_html    ( file_name )
    elif ( file_type == "pdf"  ) : df = tabula.read_pdf ( file_name )
    else : print("dont pass in garbage filetypes")

    return df


# }}}

# Def : clean_cities {{{
def clean_cities( cities , league ):
    cities = cities.rename ( columns={"Population (2016 est.)[8]": "Population"} )
    cities = cities.iloc[:-1,[0,3,5,6,7,8]]
    cities = cities[['Metropolitan area' , 'Population' , league]]
    cities = cities.replace( {r'\[([^)]+)\]' : r''    } , regex = True ) # replace [notes] with ""
    cities = cities.replace( {r'^(?![\s\S])' : r'—'   } , regex = True ) # replace "" with -
    cities = cities.replace( {r'^—$'         : np.nan } , regex = True ) # replace - with nan
    cities = cities.replace( {r'— '          : np.nan } , regex = True ) # replace - with nan
    cities = cities.dropna()                                             # drop nan

    cities["Population"] = (cities["Population"]).astype(float)          # population column ensure float type

    return cities


# }}}
# Def : clean_nhl {{{
def clean_nhl( df ):
    df = df[df['year'] == 2018]
    df = df[['team','W','L']]
    df = df.replace     ( {r'[*]{1}' : r''}  , regex = True) # remove *
    df = df.sort_values ( 'W', ascending=False)              # remove ... division , bring to top
    df = df.drop        ( df.index[:4])                      # remove ... division , drop top 4

    team_dict = dict()
    for label,data in df['team'].iteritems():
        team_dict[data] = data.split()[-1]                     # get teams without reigons

    df        = df.set_index("team") # dict needs team name as key / index
    df['NHL'] = pd.Series(team_dict) # new column with team names
    df        = df.reset_index()     # reset to not lose column

    # team names: align differences in names
    mapping = {
                'Jackets' : 'Blue Jackets'   ,
                'Leafs'   : 'Maple Leafs'    ,
                'Knights' : 'Golden Knights' ,
                'Wings'   : 'Red Wings'
            }
    df['NHL'] = df['NHL'].replace(mapping)

    # team names: information merged by metro area , so merge the relevant rows
    df = df.set_index("NHL")
    df = df.append(pd.Series( name='RangersIslandersDevils') )
    df = df.append(pd.Series( name='KingsDucks') )

    # team names : add wins and losses for merged teams
    df.loc["RangersIslandersDevils"]['W']  = float( df.loc["Rangers"]  ['W'] )
    df.loc["RangersIslandersDevils"]['W'] += float( df.loc["Islanders"]['W'] )
    df.loc["RangersIslandersDevils"]['W'] += float( df.loc["Devils"]   ['W'] )
    df.loc["RangersIslandersDevils"]['L']  = float( df.loc["Rangers"]  ['L'] )
    df.loc["RangersIslandersDevils"]['L'] += float( df.loc["Islanders"]['L'] )
    df.loc["RangersIslandersDevils"]['L'] += float( df.loc["Devils"]   ['L'] )

    df.loc["KingsDucks"]['W']  = float( df.loc["Kings"]['W'] )
    df.loc["KingsDucks"]['W'] += float( df.loc["Ducks"]['W'] )
    df.loc["KingsDucks"]['L']  = float( df.loc["Kings"]['L'] )
    df.loc["KingsDucks"]['L'] += float( df.loc["Ducks"]['L'] )

    df.loc["RangersIslandersDevils"]['team'] = "NA"
    df.loc["KingsDucks"]['team']             = "NA"

    df['W'] = (df['W']).astype(float) # win column ensure float type
    df['L'] = (df['L']).astype(float) # loss column ensure float type

    df = df.reset_index()
    #print(nhl_df)
    return df

# }}}
# Def : clean_nba {{{
def clean_nba( df ):
    df = df[df['year'] == 2018]
    df = df[['team','W','L']]
    df = df.replace( {r'[*]{1}'          : r''   } , regex = True ) # remove *
    df = df.replace( {r'\(([^)]+)\)'     : r''   } , regex = True ) # replace (...) with ""
    df = df.replace( {r'([\w]+)([\s]+$)' : r'\1' } , regex = True ) # remove trailing whitespaces

    team_dict = dict()
    for label,data in df['team'].iteritems():
        team_dict[data] = data.split()[-1]   # get teams without reigons

    df        = df.set_index("team") # dict needs team name as key / index
    df['NBA'] = pd.Series(team_dict) # new column with team names
    df        = df.reset_index()     # reset to not lose column

    # team names: align differences in names
    mapping = {
                'Blazers' : 'Trail Blazers'   ,
              }
    df['NBA'] = df['NBA'].replace(mapping)

    # team names: information merged by metro area , so merge the relevant rows
    df = df.set_index("NBA")
    df = df.append(pd.Series( name='KnicksNets') )
    df = df.append(pd.Series( name='LakersClippers') )

    # team names : add wins and losses for merged teams
    df.loc["KnicksNets"]['W']  = float( df.loc["Knicks"]['W'] )
    df.loc["KnicksNets"]['W'] += float( df.loc["Nets"]  ['W'] )
    df.loc["KnicksNets"]['L']  = float( df.loc["Knicks"]['L'] )
    df.loc["KnicksNets"]['L'] += float( df.loc["Nets"]  ['L'] )

    df.loc["LakersClippers"]['W']  = float( df.loc["Lakers"]  ['W'] )
    df.loc["LakersClippers"]['W'] += float( df.loc["Clippers"]['W'] )
    df.loc["LakersClippers"]['L']  = float( df.loc["Lakers"]  ['L'] )
    df.loc["LakersClippers"]['L'] += float( df.loc["Clippers"]['L'] )

    df.loc["KnicksNets"]    ['team'] = "NA"
    df.loc["LakersClippers"]['team'] = "NA"

    df['W'] = (df['W']).astype(float) # win column ensure float type
    df['L'] = (df['L']).astype(float) # loss column ensure float type

    df = df.reset_index()
    return df

# }}}
# Def : clean_mlb {{{

def clean_mlb( df ):
    df = df[df['year'] == 2018]
    df = df[['team','W','L']]
    df = df.replace( {r'[*]{1}'          : r''   } , regex = True ) # remove *
    df = df.replace( {r'[+]{1}'          : r''   } , regex = True ) # remove +
    df = df.replace( {r'([\w]+)([\s]+$)' : r'\1' } , regex = True ) # remove trailing whitespaces

    team_dict = dict()
    for label,data in df['team'].iteritems():
        # hack to fix 2 teams ending up with key = sox
        if   ( data == "Boston Red Sox"   ): team_dict[data] = "Red Sox"
        elif ( data == "Chicago White Sox"): team_dict[data] = "White Sox"
        else : team_dict[data] = data.split()[-1]   # get teams without reigons

    df        = df.set_index("team") # dict needs team name as key / index
    df['MLB'] = pd.Series(team_dict) # new column with team names
    df        = df.reset_index()     # reset to not lose column

    # team names: align differences in names
    mapping = {
                'Jays' : 'Blue Jays'
              }
    df['MLB'] = df['MLB'].replace(mapping)

    df = df.set_index("MLB")

    m1 = float(df.loc["Yankees"]['W']) + float(df.loc["Mets"]['W'])
    m2 = float(df.loc["Yankees"]['L']) + float(df.loc["Mets"]['L'])
    df = df.append(pd.Series( ["NA" , m1 , m2] , index = ['team','W','L'] , name = 'YankeesMets' ))

    m1 = float(df.loc["Dodgers"]['W']) + float(df.loc["Angels"]['W'])
    m2 = float(df.loc["Dodgers"]['L']) + float(df.loc["Angels"]['L'])
    df = df.append(pd.Series( ["NA" , m1 , m2] , index = ['team','W','L'] , name = 'DodgersAngels' ))

    m1 = float(df.loc["Giants"]['W']) + float(df.loc["Athletics"]['W'])
    m2 = float(df.loc["Giants"]['L']) + float(df.loc["Athletics"]['L'])
    df = df.append(pd.Series( ["NA" , m1 , m2] , index = ['team','W','L'] , name = 'GiantsAthletics' ))

    m1 = float(df.loc["Cubs"]['W']) + float(df.loc["White Sox"]['W'])
    m2 = float(df.loc["Cubs"]['L']) + float(df.loc["White Sox"]['L'])
    df = df.append(pd.Series( ["NA" , m1 , m2] , index = ['team','W','L'] , name = 'CubsWhite Sox' ))

    df['W'] = (df['W']).astype(float) # win column ensure float type
    df['L'] = (df['L']).astype(float) # loss column ensure float type

    df = df.reset_index()

    #print(df)
    return df



# }}}
# Def : clean_nfl {{{
def clean_nfl( df ):
    df = df[df['year'] == 2018]
    df = df[['team','W','L']]
    df = df.replace( {r'[*]{1}'          : r''   } , regex = True ) # remove *
    df = df.replace( {r'[+]{1}'          : r''   } , regex = True ) # remove +
    df = df.sort_values('W', ascending=False)       # remove ... division , bring to top
    df = df.drop(df.index[:8])                     # remove ... division , drop top 8

    team_dict = dict()
    for label,data in df['team'].iteritems():
        team_dict[data] = data.split()[-1]                     # get teams without reigons

    df        = df.set_index("team") # dict needs team name as key / index
    df['NFL'] = pd.Series(team_dict)     # new column with team names
    df        = df.reset_index()     # reset to not lose column

    df = df.set_index("NFL")
    df = df.append(pd.Series( name='GiantsJets') )
    df = df.append(pd.Series( name='RamsChargers') )
    df = df.append(pd.Series( name='49ersRaiders') )

    # team names : add wins and losses for merged teams
    df.loc["GiantsJets"]['W']  = float( df.loc["Jets"]  ['W'] )
    df.loc["GiantsJets"]['W'] += float( df.loc["Giants"]['W'] )
    df.loc["GiantsJets"]['L']  = float( df.loc["Jets"]  ['L'] )
    df.loc["GiantsJets"]['L'] += float( df.loc["Giants"]['L'] )

    df.loc["RamsChargers"]['W']  = float( df.loc["Rams"]['W'] )
    df.loc["RamsChargers"]['W'] += float( df.loc["Chargers"]['W'] )
    df.loc["RamsChargers"]['L']  = float( df.loc["Rams"]['L'] )
    df.loc["RamsChargers"]['L'] += float( df.loc["Chargers"]['L'] )

    df.loc["49ersRaiders"]['W']  = float( df.loc["49ers"]['W'] )
    df.loc["49ersRaiders"]['W'] += float( df.loc["Raiders"]['W'] )
    df.loc["49ersRaiders"]['L']  = float( df.loc["49ers"]['L'] )
    df.loc["49ersRaiders"]['L'] += float( df.loc["Raiders"]['L'] )

    df.loc["GiantsJets"]  ['team'] = "NA"
    df.loc["RamsChargers"]['team'] = "NA"
    df.loc["49ersRaiders"]['team'] = "NA"

    df['W'] = (df['W']).astype(float) # win column ensure float type
    df['L'] = (df['L']).astype(float) # loss column ensure float type

    df = df.reset_index()
    #print(df)
    return df


# }}}

# Def : nhl_correlation {{{
def nhl_correlation():
    cities_url  = "../../data/wikipedia_data.html"
    league_file = "../../data/nhl.csv"
    league      = 'NHL'
    cities      = df_read      ( cities_url      ) [1]
    cities      = clean_cities ( cities , league )
    league_df   = df_read      ( league_file     )
    league_df   = clean_nhl    ( league_df       )

    df                   = pd.merge(cities , league_df , how = "outer" , on = league)
    df                   = df.set_index(league)
    df                   = df.dropna()
    df["w/l"]            = df['W']/(df['W']+df['L'])
    population_by_region = df["Population"]
    win_loss_by_region   = df["w/l"]

    assert len(population_by_region) == len(win_loss_by_region), "Q1: Your lists must be the same length"
    assert len(population_by_region) == 28                     , "Q1: There should be 28 teams being analysed for NHL"

    correlation  = stats.pearsonr(population_by_region, win_loss_by_region)

    print(df)

    return correlation[0]


# }}}
# Def : nba_correlation {{{
def nba_correlation():

    cities_url  = "../../data/wikipedia_data.html"
    league_file = "../../data/nba.csv"
    league      = 'NBA'
    cities      = df_read      ( cities_url      ) [1]
    cities      = clean_cities ( cities , league )
    league_df   = df_read      ( league_file     )
    league_df   = clean_nba    ( league_df       )

    df                   = pd.merge(cities , league_df , how = "outer" , on = league)
    df                   = df.set_index(league)
    df                   = df.dropna()
    df["w/l"]            = df['W']/(df['W']+df['L'])
    population_by_region = df["Population"]
    win_loss_by_region   = df["w/l"]

    assert len(population_by_region) == len(win_loss_by_region), "Q2: Your lists must be the same length"
    assert len(population_by_region) == 28, "Q2: There should be 28 teams being analysed for NBA"

    correlation  = stats.pearsonr(population_by_region, win_loss_by_region)

    print(df)

    return correlation[0]


# }}}
# Def : mlb_correlation {{{
def mlb_correlation():
    cities_url  = "../../data/wikipedia_data.html"
    league_file = "../../data/mlb.csv"
    league      = 'MLB'
    cities      = df_read      ( cities_url      ) [1]
    cities      = clean_cities ( cities , league )
    league_df   = df_read      ( league_file     )
    league_df   = clean_mlb    ( league_df       )
    #print( cities )
    #print( league_df )
    df                   = pd.merge(cities , league_df , how = "outer" , on = league)
    df                   = df.set_index(league)
    df                   = df.dropna()
    df["w/l"]            = df['W']/(df['W']+df['L'])
    population_by_region = df["Population"]
    win_loss_by_region   = df["w/l"]

    #df["win/loss ratio"] = df['W']/(df['W']+df['L']) # Win/Loss ratio = # of wins / (# wins + # losses)
    #population_by_region = df["Population"]          # pass in metropolitan area population from cities
    #win_loss_by_region   = df["win/loss ratio"]      # pass in win/loss ratio from mlb_df in the same order as cities["Metropolitan area"]

    assert len(population_by_region) == len(win_loss_by_region), "Q3: Your lists must be the same length"
    assert len(population_by_region) == 26,                      "Q3: There should be 26 teams being analysed for MLB"

    correlation  = stats.pearsonr(population_by_region, win_loss_by_region)

    print(df)

    return correlation[0]

# }}}
# Def : nfl_correlation {{{
def nfl_correlation():
    cities_url  = "../../data/wikipedia_data.html"
    league_file = "../../data/nfl.csv"
    league      = 'NFL'
    cities      = df_read      ( cities_url      ) [1]
    cities      = clean_cities ( cities , league )
    league_df   = df_read      ( league_file     )
    league_df   = clean_nfl    ( league_df       )
    df                   = pd.merge(cities , league_df , how = "outer" , on = league)
    df                   = df.set_index(league)
    df                   = df.dropna()
    df["w/l"]            = df['W']/(df['W']+df['L'])
    population_by_region = df["Population"]
    win_loss_by_region   = df["w/l"]

    assert len(population_by_region) == len(win_loss_by_region), "Q4: Your lists must be the same length"
    assert len(population_by_region) == 29, "Q4: There should be 29 teams being analysed for NFL"

    correlation  = stats.pearsonr(population_by_region, win_loss_by_region)

    print(df)

    return correlation[0]

# }}}

if __name__ == '__main__':
    nhl_correlation()
    nba_correlation()
    mlb_correlation()
    nfl_correlation()


