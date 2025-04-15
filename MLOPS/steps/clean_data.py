import logging
import pandas as pd
from zenml import step

@step
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Your data cleaning code here
    cleaned_df = df.dropna()  # Example cleaning operation
    return cleaned_df
