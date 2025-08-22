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
            df[col] = pd.to_numeric(df[col])
        
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


class ColumnHandler(MissingValueHandlingStrategy):
    def __init__(self, new_column = "", old_column ="") :
        self.new_column = new_column
        self.old_column = old_column
        logging.info(f"Handling Columns")

    def handle(self, df):
        df[self.new_column] = (df[self.old_column] != "No").astype(int)

        return df


class FillMissingValuesStrategy(MissingValueHandlingStrategy):

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
    

