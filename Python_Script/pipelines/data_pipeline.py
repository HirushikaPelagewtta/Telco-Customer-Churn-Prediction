import os
import sys
import pandas as pd
from typing import Dict
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_ingestion import DataIngestorCSV
from handle_missing_values import  ReplaceWithZero ,YesNoToBinary, ColumnHandler
from outlier_detection import OutlierDetector, IQROutlierDetection
from feature_binning import CustomBinningStratergy
from feature_encoding import OrdinalEncodingStratergy, NominalEncodingStrategy
from feature_scaling import MinMaxScalingStratergy
from data_spiltter import SimpleTrainTestSplitStratergy
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from config import get_data_paths, get_columns, get_missing_values_config, get_outlier_config, get_binning_config, get_encoding_config, get_scaling_config, get_splitting_config

# How to Run
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# .venv\Scripts\activate


def data_pipeline(
                    data_path: str='data/raw/telco-customer-dataset.xls', 
                    target_column: str='Churn', 
                    test_size: float=0.2, 
                    force_rebuild: bool=False
                    ) -> Dict[str, np.ndarray]:
    
    data_paths = get_data_paths()
    columns = get_columns()
    outlier_config = get_outlier_config()
    binning_config = get_binning_config()
    encoding_config = get_encoding_config()
    scaling_config = get_scaling_config()
    splitting_config = get_splitting_config()
    

    print('Step 1: Data Ingestion')
    artifacts_dir = os.path.join(os.path.dirname(__file__), '..', data_paths['data_artifacts_dir'])
    x_train_path = os.path.join(artifacts_dir, 'X_train.csv') # this only creates a string
    x_test_path = os.path.join(artifacts_dir, 'X_test.csv')
    y_train_path = os.path.join(artifacts_dir, 'Y_train.csv')
    y_test_path = os.path.join(artifacts_dir, 'Y_test.csv')

    if os.path.exists(x_train_path) and \
       os.path.exists(x_test_path) and \
       os.path.exists(y_train_path) and \
       os.path.exists(y_test_path):
        
        X_train =pd.read_csv(x_train_path)
        X_test =pd.read_csv(x_test_path)
        Y_train =pd.read_csv(y_train_path)
        Y_test =pd.read_csv(y_test_path)

    os.makedirs(data_paths['data_artifacts_dir'], exist_ok=True)
    if not os.path.exists('temp_imputed.csv'):
        ingestor = DataIngestorCSV()
        df = ingestor.ingest(data_path)
        print(f"loaded data shape: {df.shape}")

        print('\nStep 2: Handle Missing Values')
        # drop_handler = DropMissingValuesStrategy(critical_columns=columns['critical_columns'])

        white_space_handler = ReplaceWithZero(critical_columns = columns['critical_columns'])

        yes_no_handler = YesNoToBinary(fill_columns = columns['fill_columns'], internet_columns = columns['internet_columns'], phone_columns = columns['phone_columns'])

        column_handler = ColumnHandler(new_column = columns['new_column'], old_column = columns['old_column'])
        # age_handler = FillMissingValuesStrategy(                
        #                                         method='mean',
        #                                         relevant_column='Age'
        #                                         )
        
        
        df = white_space_handler.handle(df)
        df = yes_no_handler.handle(df)
        df = column_handler.handle(df)

        df.to_csv('temp_imputed.csv', index=False)

    df = pd.read_csv('temp_imputed.csv')

    print(f"data shape after imputation: {df.shape}")

    print('\nStep 3: Handle Outliers')

    outlier_detector = OutlierDetector(strategy=IQROutlierDetection())
    df = outlier_detector.handle_outliers(df, columns['outlier_columns'])
    print(f"data shape after outlier removal: {df.shape}")

    print('\nStep 4: Feature Binning')

    binning = CustomBinningStratergy(binning_config['tenure_bins'])
    df = binning.bin_feature(df, 'tenure')
    print(f"data after feature binning: \n{df.head()}")

    print('\nStep 5: Feature Encoding')

    nominal_strategy = NominalEncodingStrategy(encoding_config['nominal_columns'])
    ordinal_strategy = OrdinalEncodingStratergy(encoding_config['ordinal_mappings'])

    df = nominal_strategy.encode(df)
    df = ordinal_strategy.encode(df)
    print(f"data after feature encoding: \n{df.head()}")

    print('\nStep 6: Feature Scaling')
    minmax_strategy = MinMaxScalingStratergy()
    df = minmax_strategy.scale(df, scaling_config['columns_to_scale'])
    print(f"data after feature scaling: \n{df.head()}")

    print('\nStep 7: Post Processing')
    df = df.drop(columns=["customerID"])
    print(f"data after post processing: \n{df.head()}")

    print('\nStep 8: Data Splitting')
    splitting_stratergy = SimpleTrainTestSplitStratergy(test_size=splitting_config['test_size'])
    X_train, X_test, Y_train, Y_test = splitting_stratergy.split_data(df, 'Exited')

    X_train.to_csv(x_train_path, index=False)
    X_test.to_csv(x_test_path, index=False)
    Y_train.to_csv(y_train_path, index=False)
    Y_test.to_csv(y_test_path, index=False)

    print(f"X train size : {X_train.shape}")
    print(f"X test size : {X_test.shape}")
    print(f"Y train size : {Y_train.shape}")
    print(f"Y test size : {Y_test.shape}")

data_pipeline()