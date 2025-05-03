from .model import MarianGlossTranslator
from .utils import load_data

def evaluate_model():
    test_texts, test_glosses = load_data("test.csv")
    model = MarianGlossTranslator()
    predictions = model.translate(test_texts)

    for pred, actual in zip(predictions, test_glosses):
        print(f"Predicted: {pred}\nActual: {actual}\n")

if __name__ == "__main__":
    evaluate_model()
