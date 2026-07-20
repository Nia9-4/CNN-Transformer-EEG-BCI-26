from src.model import CNN
from src.dataset import create_dataloaders
from src.train import train_model
from src.evaluate import evaluate_model

def main():

    train_loader, test_loader = create_dataloaders()

    model = CNN()

    train_model(model, train_loader)

    evaluate_model(model, test_loader)

if __name__ == "__main__":
    main()