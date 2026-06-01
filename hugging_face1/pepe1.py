from transformers import pipeline 

classifier = pipeline("sentiment-analysis")

result = classifier("i want to run the code and get  the high score")

print(result)

generator = pipeline("text-generation", model = 'distilgpt2')

res = generator("i want to run this code and get the high score", max_length = 50, num_return_sequences = 3)

print(res)