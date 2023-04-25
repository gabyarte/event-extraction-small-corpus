import pandas as pd

def rdisplay(df):
    pd.set_option('display.max_rows', None)
    display(df)

def cdisplay(df):
    pd.set_option('display.max_columns', None)
    display(df)
