import os
import pandas as pd

Ravdess = "Datasets/RAVDESS/"
Savee = "Datasets/SAVEE/"
Tess = "Datasets/TESS/"

RAVDESS_EMOTIONS = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fear",
    7: "disgust",
    8: "surprise",
}


def load_ravdess():
    ravdess_emotions = []
    ravdess_paths = []
    ravdess_directories = [
        d for d in os.listdir(Ravdess) if os.path.isdir(os.path.join(Ravdess, d))
    ]

    for directories in ravdess_directories:
        actor = os.listdir(Ravdess + directories)
        for audio in actor:
            part = audio.split(".")[0].split("-")
            ravdess_emotions.append(int(part[2]))
            ravdess_paths.append(Ravdess + directories + "/" + audio)

    df = pd.DataFrame({"Path": ravdess_paths, "Emotions": ravdess_emotions})
    df["Emotions"] = df["Emotions"].map(RAVDESS_EMOTIONS)
    return df


def load_savee():
    savee_emotions = []
    savee_paths = []
    savee_files = [
        d for d in os.listdir(Savee) if os.path.isfile(os.path.join(Savee, d))
    ]

    for file in savee_files:
        savee_paths.append(Savee + file)
        part = file.split("_")[1][:-6]
        mapping = {
            "a": "angry",
            "d": "disgust",
            "f": "fear",
            "h": "happy",
            "n": "neutral",
            "sa": "sad",
            "su": "surprise",
        }
        savee_emotions.append(mapping.get(part, "Unknown"))
    return pd.DataFrame({"Path": savee_paths, "Emotions": savee_emotions})


def load_tess():
    tess_emotions = []
    tess_paths = []
    tess_files = [d for d in os.listdir(Tess) if os.path.isfile(os.path.join(Tess, d))]

    for file in tess_files:
        tess_paths.append(Tess + file)
        emotion = file.split("_")[2][:-4]
        mapping = {
            "angry": "angry",
            "disgust": "disgust",
            "fear": "fear",
            "happy": "happy",
            "neutral": "neutral",
            "sad": "sad",
            "ps": "surprise",
        }
        tess_emotions.append(mapping.get(emotion, "Unknown"))
    return pd.DataFrame({"Path": tess_paths, "Emotions": tess_emotions})


def prepare_dataset():
    ravdess_df = load_ravdess()
    savee_df = load_savee()
    tess_df = load_tess()

    combined_df = pd.concat([ravdess_df, savee_df, tess_df], axis=0)
    combined_df = combined_df[~combined_df["Emotions"].isin(["Unknown", "calm"])]
    combined_df.drop_duplicates(inplace=True)
    combined_df.to_csv("Output/Audio Dataset.csv", index=False)
    return combined_df


if __name__ == "__main__":
    prepare_dataset()
