import pandas as pd
import os

Ravdess = "Audio_Speech_Actors_01-24/"
Crema = "AudioWAV/"

ravdess_emotions = []
ravdess_paths = []
ravdess_intensity = []
ravdess_directories = [
    d for d in os.listdir(Ravdess) if os.path.isdir(os.path.join(Ravdess, d))
]

for directories in ravdess_directories:
    actor = os.listdir(Ravdess + directories)
    for audio in actor:
        part = audio.split(".")[0]
        part = part.split("-")
        ravdess_emotions.append(int(part[2]))
        ravdess_intensity.append(int(part[3]))
        ravdess_paths.append(Ravdess + directories + "/" + audio)

Ravdess_df = pd.DataFrame(
    {
        "Path": ravdess_paths,
        "Emotions": ravdess_emotions,
        "Intensity": ravdess_intensity,
    }
)

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

RAVDESS_INTENSITY = {1: 0, 2: 2}

Ravdess_df["Emotions"] = Ravdess_df["Emotions"].map(RAVDESS_EMOTIONS)
Ravdess_df["Intensity"] = Ravdess_df["Intensity"].map(RAVDESS_INTENSITY)

crema_emotions = []
crema_paths = []
crema_intensity = []
crema_files = [d for d in os.listdir(Crema) if os.path.isfile(os.path.join(Crema, d))]

for file in crema_files:
    crema_paths.append(Crema + file)
    part = file.split("_")
    if part[2] == "SAD":
        crema_emotions.append("sad")
    elif part[2] == "ANG":
        crema_emotions.append("angry")
    elif part[2] == "DIS":
        crema_emotions.append("disgust")
    elif part[2] == "FEA":
        crema_emotions.append("fear")
    elif part[2] == "HAP":
        crema_emotions.append("happy")
    elif part[2] == "NEU":
        crema_emotions.append("neutral")
    else:
        crema_emotions.append("Unknown")
    if part[3] == "HI.wav":
        crema_intensity.append(2)
    elif part[3] == "MD.wav":
        crema_intensity.append(1)
    elif part[3] == "LO.wav":
        crema_intensity.append(0)
    else:
        crema_intensity.append(-1)
Crema_df = pd.DataFrame(
    {
        "Path": crema_paths,
        "Emotions": crema_emotions,
        "Intensity": crema_intensity
    }
)
Crema_df = Crema_df.drop(Crema_df[Crema_df['Intensity'] < 0].index, inplace=False)
Crema_df = Crema_df.drop(Crema_df[Crema_df['Emotions'] == "Unknown"].index, inplace=False)
