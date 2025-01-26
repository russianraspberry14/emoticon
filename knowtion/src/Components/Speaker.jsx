import React, { useState } from "react";

const Speaker = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [audioSrc, setAudioSrc] = useState(null);
  let mediaRecorder;
  let audioChunks = [];

  const startRecording = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);

    mediaRecorder.ondataavailable = (event) => audioChunks.push(event.data);
    mediaRecorder.onstop = () => {
      const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
      const audioUrl = URL.createObjectURL(audioBlob);
      setAudioSrc(audioUrl);

      // Send the audio to the server
      const formData = new FormData();
      formData.append("audio", audioBlob, "recording.wav");

      fetch("/upload-audio", {
        method: "POST",
        body: formData,
      })
        .then((response) => response.text())
        .then((data) => console.log(data));
    };

    mediaRecorder.start();
    setIsRecording(true);
    console.log(`MediaRecorder is in "${mediaRecorder.state}" state `);
    console.log(`MediaRecorder is in "${mediaRecorder}" `);
  };

 const stopRecording = () => {
  // Check if mediaRecorder is defined and in the "recording" state
  console.log("inside stop");
  console.log(`MediaRecorder is in "${mediaRecorder.state}" state `);
    console.log(`MediaRecorder is in "${mediaRecorder}" `);
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop(); // Stop recording
    console.log("Recording stopped.");
  } else if (!mediaRecorder) {
    console.error("MediaRecorder is not initialized.");
  } else {
    console.error(`MediaRecorder is in "${mediaRecorder.state}" state and cannot stop.`);
  }
};

  return (
    <section>
      <h1>Record Your Voice</h1>
      <button
        onClick={startRecording}
         disabled={isRecording}
      >
        🎤 Start Recording
      </button>
      <button
        onClick={stopRecording}
        disabled={!isRecording}
      >
        ⏹ Stop Recording
      </button>
      <audio src={audioSrc} controls></audio>
    </section>
  );
};

export default Speaker;
