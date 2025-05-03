from transformers import MarianTokenizer, MarianMTModel

class MarianGlossTranslator:
    def __init__(self, model_name='Helsinki-NLP/opus-mt-en-ROMANCE'):
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)

    def train(self, texts, glosses):
        # Placeholder: MarianMT is pretrained and not typically fine-tuned this way
        print("Training skipped (MarianMT is not typically fine-tuned on small datasets)")

    def translate(self, texts):
        batch = self.tokenizer.prepare_seq2seq_batch(texts, return_tensors="pt")
        translated = self.model.generate(**batch)
        return [self.tokenizer.decode(t, skip_special_tokens=True) for t in translated]
