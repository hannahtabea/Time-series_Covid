import pandas as pd
#import datetime as dt
import matplotlib.pyplot as plt

# import only variables that have importance:   
cols = ['RecordedDate','UserLanguage','Dem_age','Dem_maritalstatus', 'Dem_dependents','Dem_islolation','Scale_PSS10_UCLA_1', 'Scale_PSS10_UCLA_2', 'Scale_PSS10_UCLA_3', 'Scale_PSS10_UCLA_4', 'Scale_PSS10_UCLA_5','Scale_PSS10_UCLA_6', 'Scale_PSS10_UCLA_7','Scale_PSS10_UCLA_8','Scale_PSS10_UCLA_9','Scale_PSS10_UCLA_10','Scale_Lon_1','Scale_Lon_2','Scale_Lon_3','Corona_concerns_1', 'Corona_concerns_2', 'Corona_concerns_3','Corona_concerns_4','Corona_concerns_5']
df = pd.read_csv('COVIDiSTRESS_April_27_clean.csv',encoding='latin1', usecols= cols)

# descriptives
df.head()
df.dtypes

# handle date times
df['RecordedDate'] = pd.to_datetime(df['RecordedDate'])

# select german entries
df_ger = df[df['UserLanguage']=='DE']
df_ger["RecordedDate"].map(lambda t: t.date()).unique()

# plot histogram of dates
plt.figure(figsize=(20, 10))
ax = (df['RecordedDate'][df['RecordedDate'].dt.month == 4].groupby(df['RecordedDate'].dt.day)
                         .count()).plot(kind="bar", color='#494949')
ax.set_facecolor('#eeeeee')
ax.set_xlabel("day of the month")
ax.set_ylabel("count")
ax.set_title("Recorded answers in April")
plt.show()


