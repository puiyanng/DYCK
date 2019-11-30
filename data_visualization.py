'''
Implement generic methods for the following:
- Correlation matrix of the S&P closing price vs all other closing prices
- Charts of returns calculated in feature_engineering.py

'''





# -*- coding: utf-8 -*-
"""# README

This notebook runs on Google Colab
The Source file results **bold text.csv** should locate in the same directory of the notebook
"""

"""#Setup"""

import pandas as pd
import tqdm as tqdm
import matplotlib as plt
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import re
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 500)

"""# Functions"""

df_cols = ['Model', 'Actual', 'First Order', 'Second Order',
           'First Order (NASDAQ & Currencies Excluded)',
           'Second Order (NASDAQ & Currencies Excluded)',
           'First Order (15day Window)', 'Second Order (15day Window)',
           'First Order (NASDAQ & Currencies Excluded, 15day Window)',
           'Second Order (NASDAQ & Currencies Excluded, 15day Window)',
           'Stationary'
          ]
cv_cols = []
for i in range(2, 16):
  cv_cols.append('First Order (NASDAQ & Currencies Excluded, CV ' + str(i) + ')')



def insertDF(the_df, model_name, value, transform_name, key_col="Model"):
    if(the_df.loc[the_df[key_col] == model_name].empty):
      n_r = len(the_df)
      the_df.loc[n_r] = None
      the_df.loc[n_r, key_col] = model_name

    # print(transform_name)

    idx = the_df.loc[the_df[key_col] == model_name][transform_name].index
    the_df.loc[idx, transform_name] = value
    return  the_df

def clean_model_name(name):
  if name == 'Adaboost Classifier':
    name = 'Adaboost'
  elif name == 'AdaBoost Classifier':
    name = 'Adaboost'
  elif name == 'Random Forest Classifier':
    name = 'Random Forest'

  return  name

def exec_slice_result(df, year):
  cols = df_cols + cv_cols

  df_ori = pd.DataFrame(columns=cols)
  result = df_ori

  for r_i, r in df.iterrows():
    model_name = df.iloc[r_i]['Model']

    model_name = clean_model_name(model_name)

    # Get Value
    value = df.iloc[r_i]['%age correct']
    if isinstance(value, str):
      value = value.replace('%', '')
    value = float(value)
    if value == None:
      continue


    # df.loc[r_i, ('%age correct')] = value

    if df.iloc[r_i]['# of years'] == year:

      excluded = df.iloc[r_i]['feature excluded']
      comments = df.iloc[r_i]['comments']

      if not isinstance(excluded, str):
        continue

      cv_sig = 'CV='

      if 'NASDAQ' in excluded:
        if 'first order' in comments:
          transform_name = 'First Order (NASDAQ & Currencies Excluded)'

          if 'window' in comments:
            transform_name = 'First Order (NASDAQ & Currencies Excluded, 15day Window)'

          elif cv_sig in comments:
            idx = comments.find(cv_sig)
            cv_value = comments[idx + len(cv_sig):]
            transform_name = 'First Order (NASDAQ & Currencies Excluded, CV ' + cv_value + ')'

        else:  # elif 'second order' in comments:
          transform_name = 'Second Order (NASDAQ & Currencies Excluded)'
          if 'window' in comments:
            transform_name = 'Second Order (NASDAQ & Currencies Excluded, 15day Window)'

          elif cv_sig in comments:
            idx = comments.find(cv_sig)
            cv_value = comments[idx + len(cv_sig):]
            transform_name = 'Second Order (NASDAQ & Currencies Excluded, CV ' + cv_value + ')'

      else:  # elif 'Date in exclude':
        if 'first order' in comments:
          transform_name = 'First Order'
          if 'window' in comments:
            transform_name = 'First Order (15day Window)'

          elif cv_sig in comments:
            idx = comments.find(cv_sig)
            cv_value = comments[idx + len(cv_sig):]
            transform_name = 'First Order, CV ' + cv_value + ')'

        else:  # elif 'second order' in comments:
          transform_name = 'Second Order'
          if 'window' in comments:
            transform_name = 'Second Order (15day Window)'

          elif cv_sig in comments:
            idx = comments.find(cv_sig)
            cv_value = comments[idx + len(cv_sig):]
            transform_name = 'Second Order, CV ' + cv_value + ')'

      if 'Preliminary' in comments:
        transform_name = 'Actual'
      elif 'Stationary' in comments:
        transform_name = 'Stationary'
      # print(model_name, value, transform_name)
      insertDF(result, model_name, value, transform_name)
  return  result

def slice_result(df):
  df_5yr = exec_slice_result(df, 5)
  df_10yr = exec_slice_result(df, 10)
  return  df_5yr, df_10yr

def plot_result(df_master, series, k='Model',
                width=5, y_lower=45, y_upper=65,
                legend_x=1, legend_y=-.5, rotation=45):
  # print(k)
  # print(df_master.columns)
  chart = df_master.plot.bar(x=k, y=series, rot=0, figsize=(width,5), cmap = 'YlGn', edgecolor = 'black')
  chart.legend(loc='lower right', bbox_to_anchor=(legend_x, legend_y))
  chart.set_ylim(y_lower, y_upper)

  h_align = 'right'
  if rotation == 0:
    h_align = 'center'

  chart.set_xticklabels(chart.get_xticklabels(), rotation=rotation, horizontalalignment=h_align)
  chart.set_yticklabels(['{:,.0%}'.format(x / 100) for x in chart.get_yticks()])

  plt.show()

  return  chart


def short_name(n):
  p = re.compile('NASDAQ & Currencies Excluded')
  n = p.sub('X Features', n)

  p = re.compile('15day Window')
  n = p.sub('15d', n)
  return n


def setup():
  data = pd.read_csv('resultsX.csv')
  print('Number of Results: ' + str(len(data)))
  # df = pd.DataFrame(columns=['Model', 'Actual', 'First Order', 'Second Order', 'First Order Denoised', 'Second Order Denoised', 'Stationary'])
  _, df_10y = slice_result(data)

  df_adab = df_10y[cv_cols + ['Model']]

  df_10y = df_10y[df_cols]

  t = df_10y
  # t = t.drop(['Actual'], axis=1)

  # print(t)

  cols = t.columns
  cols = cols.to_list()
  cols.remove('Model')

  plot_result(df_10y, cols, width=12, legend_y=-.9)


  sel_cols = ["First Order", "Second Order", "First Order (NASDAQ & Currencies Excluded)",
              "Second Order (NASDAQ & Currencies Excluded)", "Stationary"]
  plot_result(df_10y, sel_cols, width=8, legend_y=-.7)

  df_arima_10y = df_10y
  df_arima_10y['ARIMA'] = np.nan

  df_arima_10y = df_arima_10y.append({'Model': 'ARIMA(2,1,2)', 'ARIMA': 54.6}, ignore_index=True)

  plot_result(df_arima_10y, sel_cols + ['ARIMA'], width=12, legend_y=-.9)

  df_arima_10y['Avg'] = np.nan
  for r_i, r in df_arima_10y.iterrows():
    df_arima_10y.loc[r_i, 'Avg'] = np.mean(r[1:])

  print(df_arima_10y)

  print('-' * 60)
  df_arima_avg_10y = df_arima_10y[["Model", "Avg"]]
  print(df_arima_avg_10y)
  plot_result(df_arima_avg_10y, ['Avg'], legend_y=0, y_upper=60)
  print('-' * 60)

  return _, df_10y, df_adab, df_arima_10y
_, df_10y, df_adab, df_arima_10y = setup()

"""# Value Differencing"""

def get_result_feature_trans(t):
  t = t.drop(['First Order (15day Window)', 'Second Order (15day Window)',
              'First Order (NASDAQ & Currencies Excluded, 15day Window)',
              'Second Order (NASDAQ & Currencies Excluded, 15day Window)',
              'Stationary'], axis=1)

  # t.to_csv('table_vd_ft.csv')

  plot_result(t, ["First Order", "Second Order"])
  plot_result(t, ["First Order (NASDAQ & Currencies Excluded)", "Second Order (NASDAQ & Currencies Excluded)"])

get_result_feature_trans(df_10y)

"""# Features Removal"""

def get_result_order_diff(t):
  plot_result(t, ["First Order", "First Order (NASDAQ & Currencies Excluded)"])
  plot_result(t, ["Second Order", "Second Order (NASDAQ & Currencies Excluded)"])

get_result_order_diff(df_10y)

"""# Stationary"""

def get_result_stationary(t):
  cols = t.columns
  cols = cols.to_list()
  cols.remove('Stationary')
  cols.remove('Model')

  plot_result(t, ['First Order', 'First Order (NASDAQ & Currencies Excluded)', 'Stationary'],
              y_lower=45, y_upper=65, legend_y=0, legend_x=2)

  # print(cols)
  t_x = t.drop(cols, axis=1)
  print(t_x)
  # t_x.to_csv('table_st_ex.csv')
  plot_result(t_x, ["Stationary"], y_lower=45, y_upper=60)

  t = t[["Model", 'First Order', 'First Order (NASDAQ & Currencies Excluded)', 'Stationary']]
  t = t[t['Model']!='LSTM Regressor']
  print(t)
  # t.to_csv('table_st.csv')
  plot_result(t, ['First Order', 'First Order (NASDAQ & Currencies Excluded)', 'Stationary'],
              y_lower=45, y_upper=65, legend_y=-.6)

get_result_stationary(df_10y)

"""# 15-day Window"""

def get_result_15day_window(t):
  t = t.drop(['Actual', 'First Order', 'Second Order',
              'First Order (NASDAQ & Currencies Excluded)',
              'Second Order (NASDAQ & Currencies Excluded)',
              'Stationary'], axis=1)
  print(t)
  # t.to_csv('table_15d_w.csv')

  plot_result(t, ["First Order (15day Window)", "Second Order (15day Window)"], y_lower=45, y_upper=60)
  plot_result(t, ["First Order (NASDAQ & Currencies Excluded, 15day Window)",
              "Second Order (NASDAQ & Currencies Excluded, 15day Window)"], y_lower=45, y_upper=60, width=6)
  plot_result(t, ["First Order (15day Window)",
              "First Order (NASDAQ & Currencies Excluded, 15day Window)"], y_lower=45, y_upper=60, width=6)
  plot_result(t, ["Second Order (15day Window)",
              "Second Order (NASDAQ & Currencies Excluded, 15day Window)"], y_lower=45, y_upper=60, width=6)
  plot_result(t, ["First Order (15day Window)", "Second Order (15day Window)",
              "First Order (NASDAQ & Currencies Excluded, 15day Window)",
              "Second Order (NASDAQ & Currencies Excluded, 15day Window)"],
              y_lower=45, y_upper=60, legend_y=-.65, width=6)

get_result_15day_window(df_10y)

"""# Feature Transformation Accuracy"""

def get_trans_accuracy(df):
  result = pd.DataFrame(columns=["Trans", "Avg"])
  for c_name in df.columns:
    if not c_name == 'Model':
      m = float(df[c_name].mean())
      result = result.append([{'Trans': c_name, 'Avg': m}])
  result = result.reset_index()
  result = result.drop(['index'], axis=1)

  print(result)

  result = result.dropna()
  # result.to_csv('table_accu.csv')

  plot_result(result, ["Avg"], k="Trans", y_lower=45, y_upper=60, legend_y=-.65, width=6)

  lgd = result

  for r_i, r in lgd.iterrows():
    trans = r['Trans']
    trans = short_name(trans)
    lgd.loc[r_i, 'Trans'] = trans

  print(">" * 60)
  print(lgd)

  plot_result(lgd, ["Avg"], k="Trans", y_lower=45, y_upper=60, legend_y=0, width=6, rotation=45)

get_trans_accuracy(df_10y)

"""# Result Transpose"""

def get_result_transform(t):
  t_t = t

  cols = t_t.columns
  cols = cols.to_list()
  cols.remove('Actual')
  t_t = t_t[cols]

  t_t = t_t.set_index('Model', inplace=False).T
  t_t = t_t.reset_index()
  t_t = t_t.rename_axis(None, axis=1)
  n = 'Model'
  t_t = t_t.rename(columns={'index': n})
  print(t_t)

  cols = t_t.columns
  cols = cols.to_list()
  cols.remove('Model')

  t_t = t_t.dropna()

  plot_result(t_t, cols, width=12, legend_y=-.8)

get_result_transform(df_10y)

"""# Print Model"""

def get_result_model(t, model):
  cols = t.columns
  cols = cols.to_list()
  cols.remove('Actual')
  cols.remove('Model')
  cols.remove('ARIMA')

  t = t[t['Model']==model]

  t_p = t[cols]

  values = t_p.iloc[0, :].tolist()

  threshold = np.mean(values)

  x = range(len(values))

  above_threshold = np.maximum(values - threshold, 0)
  below_threshold = np.minimum(values, threshold)

  ax = plot_result(t, cols, y_lower=45, y_upper=65, legend_y=0, legend_x=2.35, rotation=0)

  cmap = matplotlib.cm.get_cmap('YlGn')

  c_range = range(0, len(cols))
  c_range = np.divide(c_range, len(cols))
  c_range = c_range.tolist()
  colours = cmap(c_range)

  fig, ax = plt.subplots()
  ax.bar(x, below_threshold, .5, color=[.3,.3,.3,1], edgecolor='black')
  ax.bar(x, above_threshold, .5, color=colours, edgecolor='black',
         bottom=below_threshold)
  ax.set_ylim(45, 65)

  ax.plot([-1, 10.5], [threshold, threshold], "k--")
  ax.text(-1.5,threshold,'Mean',horizontalalignment='right', verticalalignment='center')
  ax.plot([-2.5, 9], [50, 50], "k--", color=[0,0,0,.2])
  ax.text(9.5,50,'Guess',horizontalalignment='left', verticalalignment='center', color=[0,0,0,.2])

  ax.set_xticks(x)

  short_cols = []
  for i in cols:
    short_cols.append(short_name(i))

  ax.set_xticklabels(short_cols, rotation=45, horizontalalignment='right')
  plt.show()

print(df_10y)
get_result_model(df_10y, "Adaboost")

def get_result_adab(t):
  model = "Adaboost"

  t = df_adab.dropna()
  # plot_result(t, cv_cols, width=8, legend_y=0, legend_x=1.7, rotation=0)

  cols = t.columns
  cols = cols.to_list()
  cols.remove('Model')

  t_p = t[cols]
  values = t_p.iloc[0, :].tolist()

  threshold = np.float64(values[1])
  threshold1 = np.mean(values)

  print('cv=3 ', threshold)
  print('mean ', threshold1)

  x = range(len(values))
  above_threshold = np.maximum(values - threshold, 0)
  below_threshold = np.minimum(values, threshold)

  cmap = matplotlib.cm.get_cmap('YlGn')

  c_range = range(0, len(cols))
  c_range = np.divide(c_range, len(cols))
  c_range = c_range.tolist()
  colours = cmap(c_range)

  fig, ax = plt.subplots()

  c = [.3,.3,.3,1]
  h = [.7,.4,.5,1]

  ax.bar(x, below_threshold, .5, color=[c, h, c, c, c, c, c, c, c, c, c, c, c], edgecolor='black')
  ax.bar(x, above_threshold, .5, color=colours, edgecolor='black',
         bottom=below_threshold)
  ax.set_ylim(55, 65)

  ax.plot([-1, 6.75], [threshold, threshold], "-->", color=h)
  ax.text(-1.5,threshold,'CV=3',horizontalalignment='right', verticalalignment='center', color=h)
  ax.plot([7.25, 14], [threshold1, threshold1], "<k--")
  ax.text(14.5,threshold1,'Mean',horizontalalignment='left', verticalalignment='center', color='k')
  ax.plot([-2.5, 15.5], [threshold1, threshold1], "--", color=[0,0,0,0])
  # ax.text(15.5,50,'Guess',horizontalalignment='left', verticalalignment='center', color=[0,0,0,.2])

  # fig.suptitle('CV search', fontsize=11)
  ax.set_xlabel('Adaboost')

  ax.set_xticks(x)
  cv_num_cols = range(2, 16)
  ax.set_xticklabels(cv_num_cols, rotation=0)
  # plt.savefig('adab_cv.png', dpi=300)
  plt.show()


get_result_adab(df_adab)

'''
get_result_feature_trans(df_10y)
get_result_order_diff(df_10y)
get_result_stationary(df_10y)
get_result_15day_window(df_10y)
get_trans_accuracy(df_10y)
get_result_transform(df_10y)
'''
