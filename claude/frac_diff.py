import sys
import os
import pickle
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from FinancialMachineLearning.features.fracdiff import FractionalDifferentiatedFeatures

def func_diff(data: pd.DataFrame):
    cols = ['adfStat','pVal','lags','nObs','95% conf', 'corr']
    out = pd.DataFrame(columns = cols)
    for d in np.linspace(0, 1, 21):
        try:
            df1 = data[['close']]
            df2 = FractionalDifferentiatedFeatures.fracDiff(df1, d = d, thres = 1e-5)
            corr = np.corrcoef(df1.loc[df2.index,'close'], df2['close'])[0,1]
            df2 = sm.tsa.stattools.adfuller(df2['close'], maxlag = 1, regression = 'c', autolag = None)
            out.loc[round(d, 2)] = list(df2[:4]) + [df2[4]['5%']] + [corr]
        except Exception as e:
            print(f'd: {round(d, 2)}, error: {e}')
            continue

    frac_close_ffd = FractionalDifferentiatedFeatures.fracDiff_FFD(data[['close']], 0.45)

    arima = sm.tsa.ARIMA(
        frac_close_ffd['close'], 
        order = (18,0,10), 
        trend = 'c'
    ).fit(method = 'innovations_mle')
    
    print(arima.summary())