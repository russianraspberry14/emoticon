# Knowtion - A Mental Health Tracker

## Inspiration
Mental health is often overlooked, yet it plays a crucial role in our daily lives. We wanted to create a tool that could provide users with real-time insights into their emotional and stress levels without requiring active input. The idea was to integrate technology seamlessly into everyday life, helping individuals track and understand their mental well-being.

## What it Does
**Knowtion** is a mental health tracker that monitors your stress and mood swings in the background. It processes data such as audio cues and physiological signals to provide timestamped reports on your emotional state throughout the day. This helps users identify patterns in their mood and stress levels, enabling them to make informed decisions about their mental health.

## How We Built It
- We used the **Librosa** library to analyze audio data for emotion detection.
- Incorporated datasets like **RAVDESS**, **TESS**, and **SAVEE** for speech emotion recognition, and **WESAD** for heartbeat stress detection.
- Experimented with various ML models and chose **XGBoost** for its performance and accuracy.
- Developed the user interface using **HTML**, **CSS**, and **React** for a clean and interactive experience.
- Built the backend with **Flask**, handling data processing and communication between the frontend and the ML models.

## Challenges We Ran Into
- **Dataset Compatibility:** Librosa wasn't compatible with one of our datasets, requiring us to adapt on the fly.
- **Model Training Time:** Training machine learning models, especially XGBoost, was time-consuming, slowing down iterations and testing.
- **Feature Constraints:** Some features, like stress detection from heartbeats and speech-to-text logging, didn't pan out due to time constraints.

## Accomplishments That We're Proud Of
- Successfully integrated multiple datasets for emotion and stress detection.
- Created an end-to-end pipeline, taking data from collection to user reporting.
- Built a functional and user-friendly web app within a short timeframe.

## What We Learned
- Working with audio data is more complex than anticipated.
- Datasets take up significantly more space than expected.
- Gained hands-on experience with:
  - **Audio Data Analysis** using Librosa.
  - **Boosting Techniques**, particularly XGBoost, understanding its strengths and limitations.
- Learned how to adapt quickly to technical challenges and deliver a functional product.

## What's Next for Knowtion
Our vision for Knowtion involves:
1. Embedding the app into a **wearable device**, such as a smartwatch, for continuous monitoring.
2. Tracking multiple data points:
   - Heartbeat
   - Stress levels
   - Tone and emotions
   - Detection of strong language
3. Generating **daily reports** with timestamps, providing insights into users' typical workdays.
4. Developing a **healthcare professional portal**:
   - Allowing users to share reports with their healthcare providers.
   - Facilitating collaboration between healthcare professionals for improved patient care.

While we couldn’t achieve these ambitious goals within the hackathon timeframe, we’ve made a solid start towards realizing this vision.

---
**Knowtion: Helping you understand and improve your mental well-being.**
