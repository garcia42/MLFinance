import numpy as np
import pandas as pd
import statsmodels.api as sm
import multiprocessing as mp

class FractionalDifferentiatedFeatures :
    @staticmethod
    def getWeights(d, size):
        w = [1.]
        for k in range(1, size):
            w_ = -w[-1] / k * (d - k + 1)
            w.append(w_)
        w = np.array(w[:: -1]).reshape(-1, 1)
        return w

    @staticmethod
    def getWeights_FFD(d, thres): 
        w = [1.]
        k = 1
        while abs(w[-1]) >= thres:
            w_ = -w[-1] / k * (d - k + 1)
            w.append(w_)
            k += 1
        w = np.array(w[:: -1]).reshape(-1, 1)[1:]
        return w

    @staticmethod
    def fracDiff_FFD(series, d, thres=1e-5):
        w = FractionalDifferentiatedFeatures.getWeights_FFD(d, thres)
        # w = getWeights(d, series.shape[0])
        # w=getWeights_FFD(d,thres)
        width = len(w) - 1
        df = {}
        for name in series.columns:
            seriesF = series[[name]].ffill().dropna()
            df_ = pd.Series(dtype = float)
            for iloc1 in range(width, seriesF.shape[0]):
                loc0 = seriesF.index[iloc1 - width]
                loc1 = seriesF.index[iloc1]
                if not np.isfinite(series.loc[loc1, name]):
                    continue
                df_.loc[loc1] = np.dot(w.T, seriesF.loc[loc0: loc1])[0, 0]

            df[name] = df_.copy(deep=True)
        df = pd.concat(df, axis=1)
        return df
    @staticmethod
    def fracDiff(series, d, thres=.01):
        w = FractionalDifferentiatedFeatures.getWeights(d, series.shape[0])
        w_ = np.cumsum(abs(w))
        w_ /= w_[-1]
        skip = w_[w_ > thres].shape[0]
        df = {}
        for name in series.columns:
            seriesF = series[[name]].ffill().dropna()
            df_ = pd.Series(dtype = float)
            for iloc in range(skip, seriesF.shape[0]):
                loc = seriesF.index[iloc]

                test_val = series.loc[loc, name]
                if isinstance(test_val, (pd.Series, pd.DataFrame)):
                    test_val = test_val.resample('1m').mean()
                if not np.isfinite(test_val).any():
                    continue
                try:
                    df_.loc[loc] = np.dot(w[-(iloc + 1):, :].T, seriesF.loc[:loc])[0, 0]
                except:
                    continue
            df[name] = df_.copy(deep=True)
        df = pd.concat(df, axis=1)
        return df

    @staticmethod
    def frac_diff_single(df: pd.DataFrame) -> pd.DataFrame:
        column = df.columns[0]  # Since we'll pass single columns
        try:
            if not np.isfinite(df[column].iloc[0]):
                return df
                
            cols = ['adfStat','pVal','lags','nObs','95% conf', 'corr']
            out = pd.DataFrame(columns=cols)
            
            for d in np.linspace(0, 1, 11):
                try:
                    df1 = df[[column]]
                    df2 = FractionalDifferentiatedFeatures.fracDiff_FFD(df1, d=d, thres=1e-5)
                    df2_copy = df2.copy()
                    corr = np.corrcoef(df1.loc[df2.index,column], df2[column])[0,1]
                    df2_stats = sm.tsa.stattools.adfuller(df2[column], maxlag=1, regression='c', autolag=None)
                    out.loc[round(d, 2)] = list(df2_stats[:4]) + [df2_stats[4]['5%']] + [corr]
                    
                    if out['pVal'][d] < .05:
                        print(f'{column} Stationary at {d}')
                        return df2_copy
                        
                except Exception as e:
                    print(f'Column: {column}, d: {round(d, 2)}, error: {e}')
                    continue
                    
            return df  # Return original if no stationary d found
            
        except Exception as e:
            print(f"Error processing {column}: {e}")
            return df

    @staticmethod
    def frac_diff_parallel(features: pd.DataFrame, num_threads=24):
        # Create a pool of workers
        pool = mp.Pool(processes=num_threads)
        
        # Prepare the column dataframes to process
        column_dfs = [features[[col]] for col in features.columns]
        
        # Process columns in parallel using the pool
        results = pool.map(FractionalDifferentiatedFeatures.frac_diff_single, column_dfs)
        
        # Close and join the pool
        pool.close()
        pool.join()
        
        # Combine results
        processed_features = pd.concat(results, axis=1)
        processed_features.dropna(inplace=True)
        
        return processed_features