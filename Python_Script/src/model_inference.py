import json
import logging
import os
import joblib, sys
from typing import Any, Dict, List, Optional, Tuple, Union
from feature_binning import CustomBinningStratergy
from feature_encoding import OrdinalEncodingStratergy
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from config import get_binning_config, get_encoding_config

logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelInference:
    def __init__(self, model_path):
        self.model_path = model_path
        self.encoders = {}
        self.load_model()
        self.binning_config = get_binning_config()
        self.encoding_config = get_encoding_config()


    def load_model(self):
        if not os.path.exists(self.model_path):
            raise ValueError("Can't load. File not found.")
        
        self.model = joblib.load(self.model_path)

    def load_encoders(self,encoder_dir):
        for file in os.listdir(encoder_dir):
            feature_name = file.split('_encoder.json')[0]  
            with open(os.path.join(encoder_dir, file), 'r')  as f:
                self.encoders[feature_name] = json.load(f)
    
    def preprocess_input(self, data):
        data = pd.DataFrame([data])

        for col, encoder in self.encoders.items():
            data[col] =  data[col].map(encoder)

        binning = CustomBinningStratergy(self.binning_config['tenure_bins'])
        data = binning.bin_feature(data, 'tenure')

        ordinal_strategy = OrdinalEncodingStratergy(self.encoding_config['ordinal_mappings'])
        data = ordinal_strategy.encode(data)    

        data["HasInternet"] = np.where(data["InternetService"] != "No", 1, 0)


        data = data.drop(columns=['customerID'])

        for col in data.select_dtypes(include="object").columns:
            # Option 1: Convert to category → codes (safer if you know unseen values won’t appear)
            data[col] = data[col].astype("category").cat.codes

            # Or Option 2 (if you expect only encoded numeric values by now):
            # raise ValueError(f"Unexpected object column {col} found in inference data")

            # Also make sure all columns are numeric
            data = data.apply(pd.to_numeric, errors="coerce")

        expected_features = self.model.get_booster().feature_names
        data = data[expected_features]

        return data
    

    def predict(self, data):
        pp_data = self.preprocess_input(data)
        Y_pred = self.model.predict(pp_data)
        Y_proba = float(self.model.predict_proba(pp_data)[:, 1])

        Y_pred = 'Churn' if Y_pred == 1 else 'Retain'
        Y_proba = round(Y_proba*100, 2)

        return {
                "Status": Y_pred,
                "Confidance": f"{Y_proba} %"
                }



