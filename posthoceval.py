import numpy as np
import pandas as pd
import copy
pred = pd.read_csv("datasets/Results_9July2017.csv")
actual = pd.read_csv('datasets/Fire_Incidents_New.csv',encoding = 'latin-1',dtype={'street':'str','number':'str'})
inspec = pd.read_csv('datasets/pli.csv')
build = pd.read_csv("datasets/pittdata.csv")

pred = pred.drop_duplicates(['Address'],keep = "first")

# clean actual fire
actual = actual[actual["CALL_CREATED_DATE"]>"2017-05-30"]
# correct address format
actual['incident_type'] = actual['inci_type'].astype(str).str[0]
actual['street'] = actual['street'].replace(to_replace=', PGH', value='', regex=True)
actual['street'] = actual['street'].replace(to_replace=', P', value='', regex=True)
actual['street'] = actual['street'].replace(to_replace=',', value='', regex=True)
actual['street'] = actual['street'].replace(to_replace='#.*', value='', regex=True)
actual['street'] = actual['street'].str.strip()
actual['number'] = actual['number'].str.strip()
actual['street'] = actual['street'].str.strip() +' ' +actual['st_type'].str.strip()
actual['street'] = actual['number'].str.strip() +' ' +actual['street'].str.strip()
# include only needed columns
actual = actual[["street","full.code"]]#,"CALL_TYPE_FINAL"]]
# only want code
actual["full.code"] = actual["full.code"].map(lambda x : x[:x.find("-")-1])
#remove false calls
actual = actual[actual["full.code"].map(lambda x : x[0] != "7")]

# clean inspection violation
inspec = inspec[inspec["INSPECTION_DATE"] > "2017-05-30"]
inspec['STREET_NAME'] = inspec['STREET_NUM'].apply(str).apply(lambda x : x[:-2]).str.strip() + " " + inspec['STREET_NAME'].str.strip()
inspec = inspec[["STREET_NAME","INSPECTION_DATE","INSPECTION_RESULT","VIOLATION"]]

# use building data for more analysis
build = build[(build.PROPERTYCITY == 'PITTSBURGH')]
build = build[np.isfinite(build.PROPERTYHOUSENUM)]
build["PROPERTYADDRESS"] = build["PROPERTYHOUSENUM"].apply(str).apply(lambda x : x[:-2]).str.strip() + " " + build["PROPERTYADDRESS"].str.strip()
build = build[["PROPERTYADDRESS","CLASSDESC"]]
build = build.drop_duplicates(keep = "first")

""" Up until now we have:
predict: drop dups, all non-nan
actual: call date limited, removed 7 (false alarm)
- street, full.code, call_type_final
inspection: street name clean, check non-nan, no dup
- inspect date, street name, inspection result, violation
building: no dupe, only pittsburgh
"""
# concat actual fire
temp = pd.Series.to_frame(actual.groupby("street")["full.code"].apply(list))
temp.columns = ["code"]
temp["street"] = temp.index
temp = pd.merge(actual.drop_duplicates("street"),temp,how="left",on="street")
del temp["full.code"]
actual = temp
del temp

# concat inspection
def inspecCode (s):
    if s == "Violations Found": return 1
    elif s == "Abated": return 2
    else: return 3
inspec["code"] = inspec["INSPECTION_RESULT"].apply(inspecCode)
inspec = inspec.sort_values("code")
inspec = inspec.drop_duplicates("STREET_NAME",keep="first")
del inspec["code"]

# concat buildings
def buildCode (s):
    if s == "RESIDENTIAL": return 1
    else: return 0
build["code"] = build["CLASSDESC"].apply(buildCode)
build = build.sort_values("code")
build = build.drop_duplicates("PROPERTYADDRESS",keep="first")
del build["code"]

all = pd.merge(pred,actual,how="left",left_on="Address",right_on="street")
all = pd.merge(all,inspec,how="left",left_on="Address",right_on="STREET_NAME")
all = pd.merge(all,build,how="left",left_on="Address",right_on="PROPERTYADDRESS")
del all["street"]
del all["STREET_NAME"]
del all["PROPERTYADDRESS"]

# filter out commercial buildings




#test = pd.merge(pred,actual,how="left",left_on="Address",right_on="street")
#1 ALLEGHENY AVE
#TODO: merge inspec shouldn't be to the left (or any of them)
# 0, 0.3, 0.7, 1