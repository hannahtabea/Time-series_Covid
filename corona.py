import pandas as pd
#import numpy as np
import csv as csv
#import datetime as dt
import matplotlib.pyplot as plt

# initialize list
liste = []
# read csv line by line
with open('COVIDiSTRESS_April_27_clean.csv', encoding = 'utf-8') as csvfile:
    datreader = csv.reader(csvfile, delimiter=',')
    for row in datreader:
        # remove elements not of interest
        del row[54:len(row)]
        liste.append(row[1:])
            
# set column names        
titles = liste[0]

# convert to pandas dataframe
df = pd.DataFrame(data = liste,  columns = titles)
# remove double header
df, df.columns = df[1:] , df.iloc[0]

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


