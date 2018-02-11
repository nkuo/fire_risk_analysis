import numpy as np
import pandas as pd
results = pd.read_csv('datasets/Results_9July2017.csv')
fire_pre14 = pd.read_csv('datasets/Fire_Incidents_New.csv',encoding = 'latin-1',dtype={'street':'str','number':'str'})
fire_pre14 = fire_pre14[fire_pre14['CALL_CREATED_DATE'] >'2017-05-30']
results = results.sort_values(by=['RiskScore'], ascending=False)
fire_pre14['incident_type'] = fire_pre14['inci_type'].astype(str).str[0]

fire_pre14['street'] = fire_pre14['street'].replace(to_replace=', PGH', value='', regex=True)
fire_pre14['street'] = fire_pre14['street'].replace(to_replace=', P', value='', regex=True)
fire_pre14['street'] = fire_pre14['street'].replace(to_replace=',', value='', regex=True)
fire_pre14['street'] = fire_pre14['street'].replace(to_replace='#.*', value='', regex=True)
fire_pre14['street'] = fire_pre14['street'].str.strip()
fire_pre14['number'] = fire_pre14['number'].str.strip()
fire_pre14['street'] = fire_pre14['street'].str.strip() +' ' +fire_pre14['st_type'].str.strip()
fire_pre14['street'] = fire_pre14['number'].str.strip() +' ' +fire_pre14['street'].str.strip()

results = results.drop_duplicates(['Address'],keep = "first")

results = results.sort_values(by=['Fire'], ascending=True)

fire_pre14 = pd.merge(results, fire_pre14, how = 'left', left_on = 'Address', right_on = 'street')

fire_pre14 = fire_pre14[fire_pre14['RiskScore'].notnull()]
fire_pre14['risk'] = pd.cut(fire_pre14['RiskScore'], [0,0.3, 0.7,1], labels=['low','medium','high'])

fire_pre14 = fire_pre14.sort_values(by=['inci_type'], ascending=True)
fire_pre14 = fire_pre14.drop_duplicates(['Address'] )
fire_pre14.to_csv('fire_results_AllProperties_020418.csv')

fire_pre14.to_csv('fire_results_check1.csv')
fire_pre14[(fire_pre14['incident_type'] == '1') & (fire_pre14['RiskScore'].notnull()) ].to_csv('fires_july17-jan18_020418.csv')

#==nathan
all = fire_pre14[["Address","Fire","risk","CALL_TYPE_FINAL"]]
all.columns = ["street","pred_fire","pred_risk","call"]
"""
inspec = pd.read_csv('datasets/pli.csv')
inspec = inspec[inspec["INSPECTION_DATE"] > "2017-05-30"]
inspec['STREET_NAME'] = inspec['STREET_NUM'].apply(str).apply(lambda x : x[:-2]).str.strip() + " " + inspec['STREET_NAME'].str.strip()
inspec = inspec[["STREET_NAME","INSPECTION_DATE","INSPECTION_RESULT","VIOLATION"]]
def inspecCode (s):
    if s == "Violations Found": return 1
    elif s == "Abated": return 2
    else: return 3
inspec["code"] = inspec["INSPECTION_RESULT"].apply(inspecCode)
inspec = inspec.sort_values("code")
inspec = inspec.drop_duplicates("STREET_NAME",keep="first")
del inspec["code"]
del inspec["INSPECTION_DATE"]
all = pd.merge(all,inspec,how="left",left_on="street",right_on="STREET_NAME")
del all["STREET_NAME"]
del all["VIOLATION"]
"""

all = all.replace(np.nan,"", regex=True)
all["has_call"] = all["call"].apply(lambda x : 1 if x != "" else 0)
temp = all.groupby(["pred_risk"]) # total incident per risk
temp2 = all.groupby(["pred_risk","call"]) # incident seperated by call type (to see code 100s)
temp3 = all.groupby(["pred_risk","has_call"]) # incident seperated by whether there's call