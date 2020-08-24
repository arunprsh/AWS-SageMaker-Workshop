
import warnings; warnings.simplefilter('ignore')

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from numpy import mean
import pandas as pd
import numpy as np
import argparse
import pickle 
import json
import os 


def model_fn(model_dir):
    """
    Load model created by Sagemaker training.
    """
    model = pickle.load(open(os.path.join(model_dir, 'model'), 'rb'))
    # Load caches/vectorizers/transformers here if needed
    return model


def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        # Add logic to transform incoming request or payload here if needed
        return request_body
    else:
        raise ValueError("The model only supports application/json input")


def predict_fn(input_data, model):
    X = json.loads(input_data)
    X = np.array([X]).reshape(1, -1)
    return model.predict(X)


def output_fn(prediction, content_type):
    # Add logic to transform output prediction or response here 
    out = {'prediction': prediction[0]}
    return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    args = parser.parse_args()
    
    # ------------------------- YOUR MODEL TRAINING LOGIC STARTS HERE -------------------------
    # Load data from the location specified by args.train (In this case, an S3 bucket)
    scaler = MinMaxScaler()
    
    train_data = pd.read_csv(os.path.join(args.train, 'train.csv'))
    columns = train_data.columns.tolist()
    train_data = scaler.fit_transform(train_data)
    train_df = pd.DataFrame(train_data, columns=columns)
    
    y_train = train_df['DMIndicator']
    X_train = train_df.drop('DMIndicator', axis=1)

    model = RandomForestClassifier(n_estimators=227, 
                             max_depth=10, 
                             max_features='auto', 
                             class_weight='balanced')
    model.fit(X_train, y_train)
    # Save the model to the location specified by args.model_dir
    pickle.dump(model, open(os.path.join(args.model_dir, 'model'), 'wb'))
    # ------------------------- YOUR MODEL TRAINING LOGIC STOPS HERE -------------------------
