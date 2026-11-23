from transformers import pipeline

# Task 1: Sentiment Analysis
classifier = pipeline("sentiment-analysis")
text = "I love learning open source software!"
print("=== Sentiment Analysis ===")
print("INPUT :", text)
print("OUTPUT:", classifier(text))

# Task 2: Fill Mask
unmasker = pipeline("fill-mask")
text2 = "I study at <mask> University."
print("=== Fill Mask ===")
print("INPUT :", text2)
for r in unmasker(text2):
    print(r["token_str"], round(r["score"], 4))
