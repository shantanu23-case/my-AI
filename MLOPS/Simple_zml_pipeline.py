from transformers import pipeline

# Step 3: Load a pre-trained model and tokenizer for sentiment analysis
sentiment_pipeline = pipeline("sentiment-analysis")

# Step 5: Process input data
def analyze_sentiment(text):
    result = sentiment_pipeline(text)
    return result

# Example usage
if __name__ == "__main__":
    input_text = "I love using ZML pipelines for quick and easy tasks!"
    sentiment = analyze_sentiment(input_text)
    print(sentiment)
