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
cols = ['RecordedDate','Country','Dem_age','Dem_maritalstatus', 'Dem_dependents',
        'Scale_PSS10_UCLA_1', 'Scale_PSS10_UCLA_2', 'Scale_PSS10_UCLA_3', 'Scale_PSS10_UCLA_4', 
        'Scale_PSS10_UCLA_5','Scale_PSS10_UCLA_6', 'Scale_PSS10_UCLA_7','Scale_PSS10_UCLA_8',
        'Scale_PSS10_UCLA_9','Scale_PSS10_UCLA_10','Corona_concerns_1', 'Corona_concerns_2', 
        'Corona_concerns_3','Corona_concerns_4','Corona_concerns_5']

#combine all files in the list if not done yet
while 'COVIDiSTRESS All months.csv' not in all_filenames:
        df_all = pd.concat([pd.read_csv(f, encoding='latin1', usecols= cols) for f in all_filenames ])
        #export to csv
        df_all.to_csv('COVIDiSTRESS All months.csv', index=False, encoding='latin1')

df = pd.read_csv('COVIDiSTRESS All months.csv',encoding='latin1', parse_dates=['RecordedDate'], index_col='RecordedDate')

#------------------------------------------------------------------------------
# descriptives
#------------------------------------------------------------------------------
print(df.head())
print(df.dtypes)

# select german entries
df['Country'].unique()
df_fr = df[df['Country']=='France']
df_fr.isnull().values.any()
#percent duplicates
df_fr.duplicated().mean()
df_fr = df_fr.drop_duplicates()

#------------------------------------------------------------------------------
# get total number of unique dates
#------------------------------------------------------------------------------
count_per_date = df_fr.index.map(lambda t: t.date()).value_counts().rename('count').sort_index()
print(count_per_date)
# 15 days with enough data, 2020-03-31 - 2020-04-14

# get n per month
n_per_month = df_fr.index.strftime("%B").value_counts()
print(n_per_month)

#------------------------------------------------------------------------------
# plot stress responses
#------------------------------------------------------------------------------
# assign category names
categories_stress = ['Never','Almost never', 'Sometimes', 'Fairly often', 'Very often']
categories_concerns = ['Strongly disagree', 'Disagree', 'Slightly disagree', 'Slightly agree', 'Agree', 'Strongly agree']

# replace colnames by more meaningful statements
stress_names = ['upset after unexpected event', 'unable to control life', 'nervous and stressed', 'confident to cope with own problems', 
             'things were going my way', 'not able to cope with tasks', 'able to control irritations', 'feel on top of things',
             'angry because things out of control', 'difficulties piling up']
concerns_names = ['myself', 'my family', 'my close friends', 'my country', 'other countries around the globe']

# select all columns that contain relevant responses
filter_stress_col = [col for col in df_fr if col.startswith('Scale')]
df_stress = df_fr[filter_stress_col]
filter_concerns_col = [col for col in df_fr if col.startswith('Corona')]
df_concerns = df_fr[filter_concerns_col]

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

# plot likert data
fig, ax = plot_likert(results=dict_stress, category_names=categories_stress, title='Perceived stress in France, spring 2020')
plt.show()
fig, ax = plot_likert(results=dict_concerns, category_names=categories_concerns, title='Concerns about the consequenses of the Corona virus in France, spring 2020')
plt.show()
#------------------------------------------------------------------------------
# calculate composite scores
#------------------------------------------------------------------------------

# recode statements with reverse valence
to_recode = ['Scale_PSS10_UCLA_4', 'Scale_PSS10_UCLA_5','Scale_PSS10_UCLA_7', 'Scale_PSS10_UCLA_8']
reverse_values = {1:5, 2:4, 4:2, 5:1}
recoder_dict = {f'{x}' : reverse_values for x in to_recode}
print(recoder_dict)
df_stress_recoded = df_stress.replace(recoder_dict)

# aggregate time series data for plotting
def aggregate_timeseries(dataframe):
    sum_scores = dataframe.sum(axis = 1)
    sum_df = sum_scores.to_frame()
    return(sum_df)

def daily_timeseries(series, name):
    df_daily = series.resample('D').agg(['mean', 'std']).interpolate()
    df_daily.columns = [name +"_M", name + "_SD"]
    return(df_daily)

stress_sum = aggregate_timeseries(df_stress_recoded)
concerns_sum = aggregate_timeseries(df_concerns)

# downsample to daily
df_stress_daily = daily_timeseries(stress_sum, 'Stress')
df_concerns_daily = daily_timeseries(concerns_sum, 'Concerns')

#------------------------------------------------------------------------------
# Plot time series
#------------------------------------------------------------------------------
# merge daily stress and concern data
df_combined = df_concerns_daily.join(df_stress_daily)

fig, ax = plt.subplots(2,1, figsize=(10, 10))
# prepare lineplot to show correlation in changes
ax[0].plot(df_combined.index, df_combined['Stress_M'], color = 'b')
ax[0].fill_between(df_combined.index,  df_combined['Stress_M'] -  df_combined['Stress_SD'], df_combined['Stress_M'] +  df_combined['Stress_SD'], color='b', alpha=0.2)
ax[0].plot(df_combined.index, df_combined['Concerns_M'], color = 'r')
ax[0].fill_between(df_combined.index,  df_combined['Concerns_M'] -  df_combined['Concerns_SD'], df_combined['Concerns_M'] +  df_combined['Concerns_SD'], color='r', alpha=0.2)
ax[0].set_xlabel("Date")
ax[0].set_ylabel("Sum score")
ax[0].legend(['Perceived stress','Concerns about corona'])
#  plot freq across time
ax[1].hist(df_fr.index,bins=61, edgecolor='k', color = 'grey')
ax[1].axhline(y = 100, color = 'grey', linestyle = '--')
ax[1].set_ylabel("N of respondents")
fig.suptitle("Stress level and overall concerns about corona in France, spring 2020", fontsize = 16)
plt.show()

#------------------------------------------------------------------------------
# correlation of individual and composite scores
#------------------------------------------------------------------------------
correlation = df_combined.corr()
print(correlation)
# limitation: the same individuals contribute to each of the sum scores
# when some highly anxious people respond on day one, both scales will be affected the same way 
# sample characteristics

# calculate individual correlations and wrange data first
stress_sum = stress_sum.reset_index()
concerns_sum = concerns_sum.reset_index()

stress_sum = stress_sum[~stress_sum.index.duplicated(keep='first')].rename(columns = {0:'Stress'})
concerns_sum = concerns_sum[~concerns_sum.index.duplicated(keep='first')].rename(columns = {0:'Concerns'})

# join the dataframe
sum_scores = stress_sum.join(concerns_sum, how='outer', sort=True)
sum_scores['Stress'].corr(sum_scores['Concerns'])

#------------------------------------------------------------------------------
# random walk?
#------------------------------------------------------------------------------
# get characteristics of daily stress values differences
df_stress_daily.diff().agg({'mean','std'})

# Generate function for bounded random walk
def bounded_random_walk(length, lower_bound,  upper_bound, start, end, std):
    assert (lower_bound <= start and lower_bound <= end)
    assert (start <= upper_bound and end <= upper_bound)

    bounds = upper_bound - lower_bound

    rand = (std * (np.random.random(length) - 0.5)).cumsum()
    rand_trend = np.linspace(rand[0], rand[-1], length)
    rand_deltas = (rand - rand_trend)
    rand_deltas /= np.max([1, (rand_deltas.max()-rand_deltas.min())/bounds])

    trend_line = np.linspace(start, end, length)
    upper_bound_delta = upper_bound - trend_line
    lower_bound_delta = lower_bound - trend_line

    upper_slips_mask = (rand_deltas-upper_bound_delta) >= 0
    upper_deltas =  rand_deltas - upper_bound_delta
    rand_deltas[upper_slips_mask] = (upper_bound_delta - upper_deltas)[upper_slips_mask]

    lower_slips_mask = (lower_bound_delta-rand_deltas) >= 0
    lower_deltas =  lower_bound_delta - rand_deltas
    rand_deltas[lower_slips_mask] = (lower_bound_delta + lower_deltas)[lower_slips_mask]

    return trend_line + rand_deltas

set.seed(69)
random_stress = bounded_random_walk(len(df_stress_daily), lower_bound=0, upper_bound =50, start=26.59, end=24, std=7)
df_stress_daily['random'] = random_stress 
df_stress_daily.loc[:,['Stress_M','random']].plot()
plt.show()

#------------------------------------------------------------------------------
# calculate autocorrelation and ACF
#------------------------------------------------------------------------------
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf
df_combined['Stress_M'].autocorr()
df_combined['Concerns_M'].autocorr()

# Compute the acf array of mean stress levels
acf_array = acf(df_combined['Stress_M'])
print(acf_array)

# Plot the acf function
plot_acf(df_combined['Stress_M'], alpha = 0.05)
plot_acf(df_combined['Concerns_M'], alpha = 0.05)

from statsmodels.tsa.stattools import adfuller
# Run the ADF test on the price series and print out the results
adfuller(df_combined['Stress_M'])
# small p value - suggests that data do not follow a random walk
adfuller(df_combined['Concerns_M'])



