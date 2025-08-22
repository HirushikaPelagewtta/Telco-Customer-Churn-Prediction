import groq
import logging
import pandas as pd
from enum import Enum
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel
from abc import ABC, abstractmethod
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()


class MissingValueHandlingStrategy(ABC):
    @abstractmethod
    def handle(self, df: pd.DataFrame) ->pd.DataFrame:
        pass

class DropMissingValuesStrategy(MissingValueHandlingStrategy):
    def __init__(self, critical_columns=[]):
        self.critical_columns = critical_columns 
        logging.info(f"Dropping rows with missing values in critical columns: {self.critical_columns}")

    def handle(self, df):
        df_cleaned = df.dropna(subset=self.critical_columns)
        n_dropped = len(df) - len(df_cleaned)
        logging.info(f"{n_dropped} has been dropped")
        return df_cleaned

class ReplaceWithZero(MissingValueHandlingStrategy):
    def __init__(self, critical_columns = []):
        self.critical_columns = critical_columns
        logging.info(f"Replacing White Spaces in critical columns: {self.critical_columns}")

    def handle(self, df):
        for col in self.critical_columns:
            df[col] = df[col].replace(r'^\s*$', 0, regex=True)
        
        return df

class YesNoToBinary(MissingValueHandlingStrategy):
    def __init__(self, fill_columns = [], internet_columns = [], phone_columns = []):
        self.fill_columns = fill_columns
        self.internet_columns = internet_columns
        self.phone_columns = phone_columns
        logging.info(f"Replacing Yes, No in fill columns: {self.fill_columns}, Resolving {self.internet_columns} and {self.phone_columns}")

    def handle(self, df):
        for col in self.fill_columns:
            df[col] = df[col].apply(lambda x: 1 if x=="Yes" else 0)

        for col in self.internet_columns:
            df[col] = df[col].replace('No internet service', 'No')
            df[col] = df[col].apply(lambda x: 1 if x=="Yes" else 0)
        
        for col in self.phone_columns:
            df[col] = df[col].replace('No phone service', 'No')
            df[col] = df[col].apply(lambda x: 1 if x=="Yes" else 0)

        return df

class Gender(str, Enum):
    MALE = 'Male'
    FEMALE = 'Female'


class GenderPrediction(BaseModel):
    firstname: str
    lastname: str
    pred_gender: Gender

class GenderImputer: 
    def __init__(self):
        self.groq_client = groq.Groq()

    def _predict_gender(self, firstname, lastname):
        prompt = f"""
            What is the most likely gender (Male or Female) for someone with the first name '{firstname}'
            and last name '{lastname}' ?

            Your response only consists of one word: Male or Female
            """
        response = self.groq_client.chat.completions.create(
                                                            model='llama-3.3-70b-versatile',
                                                            messages=[{"role": "user", "content": prompt}],
                                                            )
        predicted_gender = response.choices[0].message.content.strip()
        prediction = GenderPrediction(firstname=firstname, lastname=lastname, pred_gender=predicted_gender)
        logging.info(f'Predicted gender for {firstname} {lastname}: {prediction}')
        return prediction.pred_gender
    
    def impute(self, df):
        missing_gender_index = df['Gender'].isnull()
        for idx in df[missing_gender_index].index:
            first_name = df.loc[idx, 'Firstname']
            last_name = df.loc[idx, 'Lastname']
            gender = self._predict_gender(first_name, last_name)
            
            if gender:
                df.loc[idx, 'Gender'] = gender
                print(f"{first_name} {last_name} : {gender}")
            else:
                print(f"{first_name} {last_name} : No Gender Detected")

        return df
    
class FillMissingValuesStrategy(MissingValueHandlingStrategy):
    """ 
    Missing -> Mean (Age)
            -> Custom (Gender)
    """
    def __init__(
                self, 
                method='mean', 
                fill_value=None, 
                relevant_column=None, 
                is_custom_imputer=False,
                custom_imputer=None
                ):
        self.method = method
        self.fill_value = fill_value
        self.relevant_column = relevant_column
        self.is_custom_imputer = is_custom_imputer
        self.custom_imputer = custom_imputer

    def handle(self, df):
        if self.is_custom_imputer:
            return self.custom_imputer.impute(df)
        df[self.relevant_column] = df[self.relevant_column].fillna(df[self.relevant_column].mean())
        logging.info(f'Missing values filled in column {self.relevant_column}.')
        return df
    

"""
23rd

-   Model & Prediction Pipelines
-   ZenML & MLflow

"""