from load_datasets import prepare_dataset
from features import process_features
from modelling import train_model


def main():
    print("Loading datasets...")
    dataset = prepare_dataset()
    print("Extracting features...")
    process_features(dataset)
    print("Training the model...")
    train_model()


if __name__ == "__main__":
    main()
