from .model import MarianGlossTranslator
from .utils import load_data

def main():
    train_texts, train_glosses = load_data("train.csv")
    model = MarianGlossTranslator()
    model.train(train_texts, train_glosses)

if __name__ == "__main__":
    main()
