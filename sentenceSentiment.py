from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis")
data = []
while True:
    try:
        user_input = input("Enter a sentence: ")
    except EOFError:
        break
    data.append(user_input)
sentiment_result = sentiment_pipeline(data)
for sentence, sentiment_result in zip(data, sentiment_result):
    print(f"\nSentence: {sentence}\nSentiment: {sentiment_result['label']}\tConfidence:{sentiment_result['score']:.10f}")
