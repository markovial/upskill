import pandas as pd
import numpy as np
import os

# Def : file_1 {{{
def file_1():
    # file_name = "assets/Energy Indicators.xls"

    path      = "../../data/energy_indicators.xls"
    dir_name  = os.path.dirname  ( __file__       )
    file_name = os.path.join     ( dir_name, path )

    xl         = pd.ExcelFile(file_name)
    skiprows   = 17
    nrows      = 244 - skiprows
    total_rows = xl.book.sheet_by_index(0).nrows
    skipfooter = total_rows - nrows - skiprows - 1
    Energy     = pd.read_excel( file_name , skiprows = skiprows , skipfooter = skipfooter , usecols = range(2 , 6) )

    # Rename columns
    Energy.columns = ['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable']

    # Country : Remove trailing numbers
    Energy['Country'].replace({r'^(\D+)(\d+)$' : r'\1'}  , regex = True , inplace = True)

    # Country : Name changes
    Energy.replace({
        "Republic of Korea": "South Korea",
        "United States of America": "United States",
        "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
        "China, Hong Kong Special Administrative Region": "Hong Kong"
        } ,
        inplace = True
    )

    # Countries : Shorten names and remove parentheses
    Energy['Country'].replace( { r"(\w+)\s\(([^)]+)\)" : r"\1"} , regex = True , inplace = True)

    # Energy Supply : Change missing data ... to NaN values
    Energy['Energy Supply'].replace( {r'\.{3}' : np.nan} , regex = True , inplace = True)

    # Energy Supply : Change Petajoules to Gigajoules

    # Energy.groupby("Energy Supply").apply( lambda x: x * 1000000 , axis =0)
    # Energy.groupby("Energy Supply").transform( lambda x: x * 1000000 )
    # print( Energy.groupby("Energy Supply").transform( lambda x: x * 2 ))
    # print( Energy["Energy Supply"].transform( lambda x: x * 1000000 ))

    Energy["Energy Supply"] = Energy["Energy Supply"].transform( lambda x: x * 1000000 )

    # print(Energy)

    return Energy

# }}}

# Def : file_2 {{{
def file_2():

    path      = "../../data/world_bank.csv"
    dir_name  = os.path.dirname  ( __file__       )
    file_name = os.path.join     ( dir_name, path )
    cols      = ['Country Name', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014','2015']
    GDP       = pd.read_csv( file_name , skiprows = 4 , usecols = cols)

    # I know I am joining on Country later
    GDP.rename(columns={"Country Name": "Country"}, inplace = True)

    # Standardize country names
    GDP.replace({
        "Korea, Rep.": "South Korea",
        "Iran, Islamic Rep.": "Iran",
        "Hong Kong SAR, China": "Hong Kong"
    } , inplace = True)

    return GDP

# }}}

# Def : file_3 {{{
def file_3():

    path      = "../../data/scimagojr-3.xlsx"
    dir_name  = os.path.dirname  ( __file__       )
    file_name = os.path.join     ( dir_name, path )
    ScimEn    = pd.read_excel(file_name)

    return ScimEn

# }}}

# Def : answer_one {{{
def answer_one():
    Energy = file_1()
    GDP    = file_2()
    ScimEn = file_3()

    df = pd.merge     ( ScimEn , Energy , how="outer" , on = ["Country"] )
    df = pd.merge     ( df     , GDP    , how="outer" , on = ["Country"] )
    df = df.set_index ( "Country" )
    df.sort_values(by=['Rank'])
    return df[:15]

# }}}

# Def : answer_two {{{
def answer_two():
    Energy = file_1()
    GDP    = file_2()
    ScimEn = file_3()

    #print(f"Energy                 : {Energy.shape[0]}")
    #print(f"GDP                    : {GDP.shape[0]}")
    #print(f"ScimEn                 : {ScimEn.shape[0]}")

    df = pd.merge( ScimEn , Energy , how="outer" , on = ["Country"] )
    #print(f"ScimEn + Energy        : {df.shape[0]}")

    df = pd.merge( df     , GDP    , how="outer" , on = ["Country"] )
    #print(f"ScimEn + Energy + GDP  : {df.shape[0]}")

    # union of all 3 - intersection of all 3
    # outer join 3 - inner join 3

    Energy = file_1()
    GDP    = file_2()
    ScimEn = file_3()
    df2 = pd.merge( ScimEn , Energy , how="inner" , on = ["Country"] )
    df2 = pd.merge( df2    , GDP    , how="inner" , on = ["Country"] )

    #print(df.shape[0] - df2.shape[0])

    return int(df.shape[0] - df2.shape[0])


# }}}

# Def : answer_three {{{
def answer_three():
    Energy = file_1()
    GDP    = file_2()
    ScimEn = file_3()

    df = pd.merge     ( ScimEn , Energy , how="inner" , on = ["Country"] )
    df = pd.merge     ( df     , GDP    , how="inner" , on = ["Country"] )

    df = df.set_index ( "Country" )
    cols_to_keep = ['2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015'];
    df = df[cols_to_keep]
    # print(df.groupby("Country").mean(numeric_only=True))
    avgGDP_dict = dict()
    for group , frame in df.groupby("Country"):
        avgGDP_dict[group] = np.nanmean(frame)
        # print(f"{group} , {np.nanmean(frame)}")

    # print(avgGDP_dict)
    avgGDP = pd.Series(avgGDP_dict)
    avgGDP.sort_values(ascending=False , inplace = True)

    print(avgGDP[:15])

    return avgGDP[:15]

# }}}

if __name__ == '__main__':
    answer_three()



