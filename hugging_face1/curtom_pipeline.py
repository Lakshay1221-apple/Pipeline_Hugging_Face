from transformers import Pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch 

class MovieGenrePipeline(Pipeline):
    
    def _sanitize_parameters(self, **kwargs):
        return {}, {}, {}
    
    def preprocess(self, text):
        return self.tokenizer(text, return_tensors = 'pt' , truncation = True, padding = True)
    
    def _forward(self, model_inputs):
        with torch.no_grad():
            return self.model(**model_inputs)
    
    def postprocess(self, model_outputs):
        probabilities = model_outputs.logits.softmax(-1).squeeze()
        labels = self.model.config.id2label
        best_class_id = probabilities.argmax().item()

        return {"label": labels[best_class_id],
                 "score": probabilities[best_class_id].item()}
    
model_id = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

model.config.id2label = {0: "Drama", 1: "Action"}

classifier = MovieGenrePipeline(model=model, tokenizer=tokenizer)

plot = classifier("A retired detective is forced back into the game when a serial killer targets his family.")
print(plot)

