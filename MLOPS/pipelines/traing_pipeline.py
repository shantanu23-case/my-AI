from zenml import pipeline
from steps.ingest import ingest_data
from steps.clean_data import clean_data
from steps.model_train import train_model
from steps.evaluate import evaluate_model  # type: ignore

@pipeline()
def training_pipeline(data_path: str):
    # Ingesting the data
    df = ingest_data(data_path=data_path)
    # Cleaning the data
    clean_df = clean_data(df=df)
    # Training the model
    model = train_model(df=clean_df)
    # Evaluating the model
    evaluate_model(model=model)