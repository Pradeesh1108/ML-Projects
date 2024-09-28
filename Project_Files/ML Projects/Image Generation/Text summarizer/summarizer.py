from transformers import T5Tokenizer, T5ForConditionalGeneration

class TextSummarizer:
    def __init__(self):
        # Load pre-trained T5 model and tokenizer
        self.model = T5ForConditionalGeneration.from_pretrained('t5-small')
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')

    def summarize(self, text, max_length=150, min_length=50):
        # Preprocess input text
        preprocess_text = text.strip().replace("\n", "")
        inputs = self.tokenizer.encode("summarize: " + preprocess_text, return_tensors="pt", max_length=512, truncation=True)

        # Generate summary
        summary_ids = self.model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
