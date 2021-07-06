import pandas as pd
from sklearn import model_selection
import config

if __name__ == '__main__':

    df = pd.read_csv(config.DATA)
    df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0) 
    
    df['kfold'] = -1
    
    df = df.sample(frac=1).reset_index(drop=True)   
    y = df['sentiment'].values
    kf = model_selection.StratifiedKFold(n_splits=config.N_SPLITS)
    
    for f, (_t, _v) in enumerate(kf.split(X=df, y=y)):
        df.loc[_v, 'kfold'] = f
        
    df.to_csv(config.DATA_FOLDS)