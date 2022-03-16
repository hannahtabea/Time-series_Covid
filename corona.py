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
cols = ['RecordedDate','UserLanguage','Dem_age','Dem_maritalstatus', 'Dem_dependents',
        'Scale_PSS10_UCLA_1', 'Scale_PSS10_UCLA_2', 'Scale_PSS10_UCLA_3', 'Scale_PSS10_UCLA_4', 
        'Scale_PSS10_UCLA_5','Scale_PSS10_UCLA_6', 'Scale_PSS10_UCLA_7','Scale_PSS10_UCLA_8',
        'Scale_PSS10_UCLA_9','Scale_PSS10_UCLA_10','Corona_concerns_1', 'Corona_concerns_2', 
        'Corona_concerns_3','Corona_concerns_4','Corona_concerns_5']

#combine all files in the list
df = pd.concat([pd.read_csv(f, encoding='latin1', usecols= cols) for f in all_filenames ])
#export to csv
df.to_csv('COVIDiSTRESS All months.csv', index=False, encoding='latin1')
df = pd.read_csv('COVIDiSTRESS All months.csv',encoding='latin1', parse_dates=['RecordedDate'])

#------------------------------------------------------------------------------
# descriptives
print(df.head())
print(df.dtypes)

# select german entries
df_ger = df[df['UserLanguage']=='DE']
df_ger.isnull().values.any()

#------------------------------------------------------------------------------
# get number of unique dates
len(df_ger["RecordedDate"].map(lambda t: t.date()).unique())
# get number of unique days per month
unique_days = (df_ger["RecordedDate"].dt.day
              .groupby(df_ger["RecordedDate"].dt.month)
              .nunique()
              .rename_axis(['month'])
              .reset_index(name='unique days'))
print(unique_days)
# get observations by month and day
n_day_month = (df_ger["RecordedDate"]   
               .groupby([df_ger["RecordedDate"].dt.month,df_ger["RecordedDate"].dt.day])
               .count()).rename_axis(['month','day'])
# pd.set_option('display.max_rows', None)
print(n_day_month)

#------------------------------------------------------------------------------
#  plot observations across time
plt.hist(df_ger['RecordedDate'],bins=61, edgecolor='k')
plt.title("Number of observations from March to June 2020")
plt.xticks(rotation = 45)
plt.savefig("Dates_hist.png", dpi=100,bbox_inches='tight')
plt.show()

#------------------------------------------------------------------------------
# plot stress responses
category_names = ['Never','Almost never', 'Sometimes', 'Fairly often', 'Very often']
# select all columns that contain stress responses
filter_col = [col for col in df_ger if col.startswith('Scale')]
df_stress = df_ger[filter_col]
# calculate percent per response
df['sales'] / df.groupby('state')['sales'].transform('sum')
# replace colnames by more meaningful statements
new_names = ['upset after unexpected event', 'unable to control life', 'nervous and stressed', 'confident to cope with own problems', 
             'things were going my way', 'not able to cope with tasks', 'able to control irritations', 'feel on top of things',
             'angry because things out of control', 'difficulties piling up']

results = pd.DataFrame(columns=new_names)
for i in range(0,len(df_stress.columns)):
    this_column = results.columns[i]
    results[this_column] = df_stress.iloc[:, i].groupby(df_stress.iloc[:, i]).count().div(len(df_stress)).mul(100).round().astype(int).reset_index(drop=True).to_frame()
dict_stress = results.to_dict('list')


def plot_likert(results, category_names, title):
    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*. The order is assumed
        to be from 'Strongly disagree' to 'Strongly aisagree'
    category_names : list of str
        The category labels.
    """
    
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    middle_index = data.shape[1]//2
    offsets = data[:, range(middle_index)].sum(axis=1) + data[:, middle_index]/2
    
    # Color Mapping
    category_colors = plt.get_cmap('coolwarm_r')(
        np.linspace(0.15, 0.85, data.shape[1]))
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot Bars
    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths - offsets
        rects = ax.barh(labels, widths, left=starts, height=0.5,
                        label=colname, color=color)
    
    # Add Zero Reference Line
    ax.axvline(0, linestyle='--', color='black', alpha=.25)
    
    # X Axis
    ax.set_xlim(-90, 90)
    ax.set_xticks(np.arange(-90, 91, 10))
    ax.xaxis.set_major_formatter(lambda x, pos: str(abs(int(x))))
    
    # Y Axis
    ax.invert_yaxis()
    
    # Remove spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Ledgend
    ax.legend(ncol=len(category_names), fontsize='small', 
              loc = 'upper center', bbox_to_anchor=(0.5, -0.05))
    
    # Title
    ax.set_title(title)
    
    # Set Background Color
    fig.set_facecolor('#FFFFFF')

    return fig, ax

fig, ax = plot_likert(dict_stress, category_names,'Perceived stress in Germany, spring 2020')
plt.show()

#To Dos
###
# have relaxations in corona restrictions lead to an decrease in psychological stress in the german population during the early corona outbreak in 2020?

# create similar plot for corona concerns 
# calculate sum scores for each participant per scale, find the relevant columns in dataset with pandas
# aggregation: compute rolling 7-day mean scores for stress + pct_change (normalize data so that it can be read as percentage change)

# add on: 
# assign age groups (18-34 young, 35-50 working age , 51-69 senior,  70-87 elderly)
# summarize corona concerns and stress by age (hypothesis: negative linear relationship between age and stress, independent from recorded date )
# + by gender (hypothesis: women are more prone to covid-induced stress)
# plot average by age group / gender

