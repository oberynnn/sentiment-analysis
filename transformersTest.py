from transformers import pipeline

# Initialize the sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Test data: a list of sentences
data = ["I love you", "I hate you"]

# Perform sentiment analysis
results = sentiment_pipeline(data)

# Print the results
for sentence, result in zip(data, results):
    print(f"Sentence: '{sentence}'\nSentiment: {result['label']}, Confidence: {result['score']:.4f}\n")

# Check if the results are as expected (optional assertion check)
expected_labels = ['POSITIVE', 'NEGATIVE']
for result, expected_label in zip(results, expected_labels):
    assert result['label'] == expected_label, f"Expected {expected_label} but got {result['label']}"

print("Test completed successfully.")
