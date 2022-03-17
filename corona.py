import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import numpy as np

#------------------------------------------------------------------------------
# set directory and handle files
#------------------------------------------------------------------------------
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
df = pd.read_csv('COVIDiSTRESS All months.csv',encoding='latin1', parse_dates=['RecordedDate'], index_col='RecordedDate')

#------------------------------------------------------------------------------
# descriptives
#------------------------------------------------------------------------------
print(df.head())
print(df.dtypes)

# select german entries
df_ger = df[df['UserLanguage']=='DE']
df_ger.isnull().values.any()

#------------------------------------------------------------------------------
# get total number of unique dates
count_per_date = df_ger.index.map(lambda t: t.date()).value_counts().rename_axis('date').reset_index(name='count')
print(count_per_date)

# get n per month
n_per_month = df_ger.index.strftime("%B").value_counts()
print(n_per_month)

#------------------------------------------------------------------------------
#  plot observations across time
plt.hist(df_ger.index,bins=61, edgecolor='k')
plt.title("Number of observations from March to June 2020")
plt.xticks(rotation = 45)
plt.savefig("Dates_hist.png", dpi=100,bbox_inches='tight')
plt.show()

#------------------------------------------------------------------------------
# plot stress responses
#------------------------------------------------------------------------------
# assign category names
categories_stress = ['Never','Almost never', 'Sometimes', 'Fairly often', 'Very often']
categories_concerns = ['Strongly disagree', 'Disagree', 'Slightly disagree', 'Slightly agree', 'Agree', 'Strongly agree']

# select all columns that contain relevant responses
filter_stress_col = [col for col in df_ger if col.startswith('Scale')]
df_stress = df_ger[filter_stress_col]
filter_concerns_col = [col for col in df_ger if col.startswith('Corona')]
df_concerns = df_ger[filter_concerns_col]

# replace colnames by more meaningful statements
stress_names = ['upset after unexpected event', 'unable to control life', 'nervous and stressed', 'confident to cope with own problems', 
             'things were going my way', 'not able to cope with tasks', 'able to control irritations', 'feel on top of things',
             'angry because things out of control', 'difficulties piling up']
concerns_names = ['myself', 'my family', 'my close friends', 'my country', 'other countries around the globe']

# define a function that takes a subset of the data and column names as input to compute percentage of chosen response categories for all columns
def create_percentages(data, cols):
    results = pd.DataFrame(columns = cols)
    for col in range(0,len(data.columns)):
        this_column = results.columns[col]
        results[this_column] = data.iloc[:, col].groupby(data.iloc[:, col]).count().div(len(data)).mul(100).round().astype(int).reset_index(drop=True).to_frame() 
    return(results.to_dict('list'))
dict_stress = create_percentages(data=df_stress, cols=stress_names)
dict_concerns = create_percentages(data=df_concerns, cols=concerns_names)

# define a funtion to plot likert data as stacked divergent bar chart
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
    # calculate offsets depending on whether number of response categories is even or odd
    if (len(category_names) % 2) == 0:
        offsets = data[:, range(middle_index)].sum(axis=1)  
    else: 
        offsets = data[:, range(middle_index)].sum(axis=1)  + data[:, middle_index]/2
    
    # Color Mapping
    category_colors = plt.get_cmap('coolwarm_r')(
        np.linspace(0.15, 0.85, data.shape[1]))
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot Bars
    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths - offsets
        ax.barh(labels, widths, left=starts, height=0.5,
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

# plot
fig, ax = plot_likert(results=dict_stress, category_names=categories_stress, title='Perceived stress in Germany, spring 2020')
plt.show()
fig, ax = plot_likert(results=dict_concerns, category_names=categories_concerns, title='Concerns about the consequenses of the Corona virus in Germany, spring 2020')
plt.show()

#To Dos
###
# have relaxations in corona restrictions lead to an decrease in psychological stress in the german population during the early corona outbreak in 2020?

# set datetime to index and adjust descriptives and histogram
# calculate sum scores for each participant per scale, find the relevant columns in dataset with pandas
# aggregation: compute rolling 7-day mean scores for stress + pct_change (normalize data so that it can be read as percentage change)

# add on: 
# assign age groups (18-34 young, 35-50 working age , 51-69 senior,  70-87 elderly)
# summarize corona concerns and stress by age (hypothesis: negative linear relationship between age and stress, independent from recorded date )
# + by gender (hypothesis: women are more prone to covid-induced stress)
# plot average by age group / gender

