import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import numpy as np

# set directory
#os.chdir("/mydir")
os.chdir("C:/Users/hanna/Dropbox/Methods_2019_2021/Python/Time series")
# list all csv files
all_filenames = [i for i in glob.glob('*.{}'.format('csv'))]
# import only  that are relevant
cols = ['RecordedDate','UserLanguage','Dem_age','Dem_maritalstatus', 'Dem_dependents','Scale_PSS10_UCLA_1', 'Scale_PSS10_UCLA_2', 'Scale_PSS10_UCLA_3', 'Scale_PSS10_UCLA_4', 'Scale_PSS10_UCLA_5','Scale_PSS10_UCLA_6', 'Scale_PSS10_UCLA_7','Scale_PSS10_UCLA_8','Scale_PSS10_UCLA_9','Scale_PSS10_UCLA_10','Corona_concerns_1', 'Corona_concerns_2', 'Corona_concerns_3','Corona_concerns_4','Corona_concerns_5']

#combine all files in the list
df = pd.concat([pd.read_csv(f, encoding='latin1', usecols= cols) for f in all_filenames ])
#export to csv
df.to_csv('COVIDiSTRESS All months.csv', index=False, encoding='latin1')
df = pd.read_csv('COVIDiSTRESS All months.csv',encoding='latin1')

# descriptives
print(df.head())
print(df.dtypes)

# handle date times
df['RecordedDate'] = pd.to_datetime(df['RecordedDate'])

# select german entries
df_ger = df[df['UserLanguage']=='DE']
# get number of unique dates
len(df_ger["RecordedDate"].map(lambda t: t.date()).unique())
# get number of unique days per month
n_days = (df_ger["RecordedDate"].dt.day.groupby( df_ger["RecordedDate"].dt.month)
  .nunique()
  .rename_axis(['month'])
  .reset_index(name='unique days'))
print(n_days)

#  plot observations across time
plt.hist(df_ger['RecordedDate'],bins=61, edgecolor='k', color = 'm')
plt.title("Number of observations from March to June 2020")
plt.xticks(rotation = 45)
plt.savefig("Dates_hist.png", dpi=100,bbox_inches='tight')
plt.show()

