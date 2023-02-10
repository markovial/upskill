import pandas as pd
import os


def get_data():
    dirname  = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'data/NISPUF17.csv')
    df       = pd.read_csv(filename)
    return df

def clean_labels(df):
    df = df.rename(mapper = str.strip          , axis = 1) ; # print(df.columns) # Column Labels : Remove trailing whitespace
    df = df.rename(mapper = lambda x:x.upper() , axis = 1) ; # print(df.columns) # Column Labels : Change all to uppercase
    return df

def proportion_of_education():
    df    = get_data()
    df    = clean_labels(df)
    df2   = df["EDUC1"]
    total = df2.shape[0] # = len(df2)
    s1    = (df2.where(df2 == 1 ).dropna()).shape[0] # 1 : <12 years
    s2    = (df2.where(df2 == 2 ).dropna()).shape[0] # 2 : 12 years
    s3    = (df2.where(df2 == 3 ).dropna()).shape[0] # 3 : >12 years, not a college graduate
    s4    = (df2.where(df2 == 4 ).dropna()).shape[0] # 4 : College graduate
    # print  ( f"{s1},{s2},{s3},{s4},{total}"  )

    proportions = {
        "less than high school"                 : (s1/total),
        "high school"                           : (s2/total),
        "more than high school but not college" : (s3/total),
        "college"                               : (s4/total)
    }
    # print (proportions)

    return proportions
def average_influenza_doses():
    df    = get_data()
    df    = clean_labels(df)

    # CBF_01   : Received breastmilk : 1 Yes , 2 No
    # P_NUMFLU : Influenza Vaccine   :
    cols  = ["CBF_01","P_NUMFLU"]
    df2   = df[cols]
    total = df2.shape[0]
    s1 = df2[
            ( df2["CBF_01"] == 1 ) # + breast milk
        ].dropna()

    s2 = df2[
            ( df2["CBF_01"] == 2 ) # - breast milk
        ].dropna()


    # print(s1.sum()[1]/s1.shape[0])
    # print(s2.sum()[1]/s2.shape[0])
    print(s1)
    print(s2)
    print(f"Sum   : {s1.sum()[1]},{s2.sum()[1]}")
    print(f"Shape : {s1.shape},{s2.shape}")


    s1_ratio = s1.sum()[1]/s1.shape[0]
    s2_ratio = s2.sum()[1]/s2.shape[0]
    print(s1_ratio)
    print(s2_ratio)
    # tuple  : (+ milk flu avg , - milk flu avg )
    return (s1_ratio,s2_ratio)

def chickenpox_by_sex():
    df    = get_data()
    df    = clean_labels(df)
    # P_NUMVRC : Vaccinated against pox, > 1
    # HAD_CPOX : Did child have pox , 1 Yes, 2 No, 3 Don’t know, 4 Refused, 5 Missing
    # SEX      : 1 Male , 2 Female
    cols  = ["HAD_CPOX","P_NUMVRC","SEX"]
    df2   = df[cols]
    total = df2.shape[0]

    pm1 = df2[
            ( df2["P_NUMVRC"] >= 1 ) & # + vax
            ( df2["HAD_CPOX"] == 1 ) & # + pox
            ( df2["SEX"]      == 1 )   # + male
        ].shape[0]
    pm2 = df2[
            ( df2["P_NUMVRC"] >= 1 ) & # + vax
            ( df2["HAD_CPOX"] == 2 ) & # - pox
            ( df2["SEX"]      == 1 )   # + male
        ].shape[0]

    pf1 = df2[
            ( df2["P_NUMVRC"] >= 1 ) & # + vax
            ( df2["HAD_CPOX"] == 1 ) & # + pox
            ( df2["SEX"]      == 2 )   # + female
        ].shape[0]
    pf2 = df2[
            ( df2["P_NUMVRC"] >= 1 ) & # + vax
            ( df2["HAD_CPOX"] == 2 ) & # - pox
            ( df2["SEX"]      == 2 )   # + female
        ].shape[0]

    # ratio : (+ vaccinated + chickenpox / + vaccinated - chicken pox)
    male_vax_pox_ratio = pm1/pm2
    female_vax_pox_ratio = pf1/pf2

    my_dict = {
        "male"   : male_vax_pox_ratio,
        "female" : female_vax_pox_ratio
    }
    return my_dict

def corr_chickenpox():
    import scipy.stats as stats
    import numpy as np
    import pandas as pd
    # this is just an example dataframe
    #df=pd.DataFrame({"had_chickenpox_column":np.random.randint(1,3,size=(100)),
    #               "num_chickenpox_vaccine_column":np.random.randint(0,6,size=(100))})

    # here is some stub code to actually run the correlation
    #corr, pval=stats.pearsonr(df["had_chickenpox_column"],df["num_chickenpox_vaccine_column"])

    # just return the correlation
    #return corr

    # YOUR CODE HERE
    #raise NotImplementedError()
    df    = get_data()
    df    = clean_labels(df)
    cols  = ["HAD_CPOX","P_NUMVRC"]
    df2   = df[cols].dropna()
    df2   = df2[ ( df2["HAD_CPOX"] == 1 ) | ( df2["HAD_CPOX"] == 2 ) ].dropna()
    # df2   = df2[ ( df2["HAD_CPOX"].isin([1,2]) ) ].dropna()
    df2.sort_index(inplace=True)

    # print(len(df2))
    corr, pval=stats.pearsonr(df2["HAD_CPOX"],df2["P_NUMVRC"])
    # print(corr)
    return corr


if __name__ == '__main__':
    # proportion_of_education()
    # average_influenza_doses()
    # chickenpox_by_sex()
    corr_chickenpox()
