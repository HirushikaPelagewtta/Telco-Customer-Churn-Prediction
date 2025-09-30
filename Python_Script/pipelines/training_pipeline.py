import os
import sys
import logging
import pandas as pd
import joblib
from typing import Dict, Any, Tuple, Optional
from data_pipeline import data_pipeline
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


from model_building import XGBoostModelBuilder, RnadomForestModelBuilder
from model_training import ModelTrainer
from model_evaluation import ModelEvaluator
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from mlflow_utils import MLflowTracker, setup_mlflow_autolog, create_mlflow_run_tags
from config import get_model_config, get_data_paths

logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def training_pipeline(
                        data_path: str='data/raw/ChurnModelling.csv', 
                        model_params: Optional[Dict[str, Any]] = None, 
                        test_size: float=0.2, 
                        random_state: int = 42,
                        model_path: str='artifacts/models/churn_analysis.joblib'
                        ):
    if (not os.path.exists(get_data_paths()['X_train'])) or \
        (not os.path.exists(get_data_paths()['Y_train'])) or \
        (not os.path.exists(get_data_paths()['X_test'])) or \
        (not os.path.exists(get_data_paths()['Y_test'])):

        data_pipeline()

    else:
        print("Loading Data Artifacrs from Data Pipelines.")

    mlflow_tracker = MLflowTracker()
    setup_mlflow_autolog()
    run_tags = create_mlflow_run_tags('training_pipeline',{
                                        'model_type': 'XGBoost',
                                        'training_strategy': 'Simple',
                                        }
                                      )
    run = mlflow_tracker.start_run(run_name='training_pipeline', tags = run_tags)
    
    X_train = pd.read_csv(get_data_paths()['X_train'])
    Y_train =  pd.read_csv(get_data_paths()['Y_train'])
    X_test =  pd.read_csv(get_data_paths()['X_test'])
    Y_test =  pd.read_csv(get_data_paths()['Y_test'])

    model_builder = XGBoostModelBuilder(**model_params)
    model = model_builder.build_model()
    
    trainer = ModelTrainer()
    model, _ = trainer.train(model = model,
                                        X_train = X_train,
                                        Y_train = Y_train.squeeze())
    
    trainer.save_model(model, model_path)

    evaluator = ModelEvaluator(model, 'XGBoost')
    evaluation_results = evaluator.evaluate(X_test, Y_test)
    evaluation_results_cp = evaluation_results.copy()
    del evaluation_results_cp['confusion_matrix']

    model_params = get_model_config()['model_params']
    mlflow_tracker.log_training_metrics(model,evaluation_results_cp, model_params)
    mlflow_tracker.end_run()
    


if __name__ == '__main__':
    model_config = get_model_config()
    model_params=model_config.get('model_params')
    training_pipeline(model_params=model_params)