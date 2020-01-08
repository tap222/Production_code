import os
import pandas as pd
#from sklearn import ensemble
from sklearn import preprocessing
from tqdm import tqdm
from sklearn import metrics
import joblib

from . import dispatcher

#import joblib

TRAINING_DATA = os.environ.get("TRAINING_DATA")
FOLD = int(os.environ.get("FOLD"))
MODEL = os.environ.get("MODEL")

FOLD_MAPPPING = {
    0: [1, 2, 3, 4],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [0, 1, 2, 3]
}

if __name__ == "__main__":
    df = pd.read_csv(TRAINING_DATA)
    #print(FOLD)
    train_df = df[df.kfold.isin(FOLD_MAPPPING.get(FOLD))].reset_index(drop=True)
    valid_df = df[df.kfold==FOLD].reset_index(drop=True)
    
    ytrain = train_df.target.values
    yvalid = valid_df.target.values
	
    train_df = train_df.drop(["id","target","kfold"], axis=1)
    valid_df = valid_df.drop(["id","target","kfold"], axis=1)
    
    valid_df = valid_df[train_df.columns]
    
    label_encoders = []
    
    for cc in tqdm(train_df.columns):
        train_df[cc] = train_df[cc].fillna(train_df[cc].mode()[0])
        valid_df[cc] = valid_df[cc].fillna(valid_df[cc].mode()[0])
    
    for c in tqdm(train_df.columns):
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train_df[c].values) + list(valid_df[c].values))
        train_df.loc[:,c] = lbl.transform(train_df[c].values)
        valid_df.loc[:,c] = lbl.transform(valid_df[c].values)
        label_encoders.append((c, lbl))
		
    #data is ready to train
    clf = dispatcher.MODELS[MODEL]
    clf.fit(train_df,ytrain)
	
    preds = clf.predict_proba(valid_df)[:,1]
    print(preds)
    print(metrics.roc_auc_score(yvalid, preds))
    
    joblib.dump(label_encoders, f"models/{MODEL}_{FOLD}_label_encoder.pkl")
    joblib.dump(clf, f"models/{MODEL}_{FOLD}.pkl")
    joblib.dump(train_df.columns, f"models/{MODEL}_{FOLD}_columns.pkl")
    