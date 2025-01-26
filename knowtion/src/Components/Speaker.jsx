// import React, { useState, useRef } from "react";

// const Speaker = () => {
//   const [isRecording, setIsRecording] = useState(false);
//   const [audioSrc, setAudioSrc] = useState(null);
//   const mediaRecorderRef = useRef(null); // Use useRef for mediaRecorder
//   const audioChunksRef = useRef([]); // Use useRef for audioChunks
//   const [predictedEmotion, setPredictedEmotion] = useState(null);

//   const startRecording = async () => {
//     try {
//       const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  
//       // Use a supported MIME type like "audio/webm"
//       const options = { mimeType: "audio/webm" };
//       const mediaRecorder = new MediaRecorder(stream, options);
  
//       // Store mediaRecorder in the useRef variable
//       mediaRecorderRef.current = mediaRecorder;
  
//       // Reset audioChunks
//       audioChunksRef.current = [];
  
//       // Push recorded data chunks into audioChunks
//       mediaRecorder.ondataavailable = (event) => {
//         audioChunksRef.current.push(event.data);
//       };
  
//       // When the recording stops
//       mediaRecorder.onstop = () => {
//         // Combine all chunks into a single Blob in WebM format
//         const audioBlob = new Blob(audioChunksRef.current, { type: "audio/webm" });
//         const audioUrl = URL.createObjectURL(audioBlob);
//         setAudioSrc(audioUrl); // Set the audio URL for playback
  
//         // Send the audio Blob to the server
//         const formData = new FormData();
//         formData.append("audio", audioBlob, "recording.webm");
  
//         fetch("http://localhost:5001/upload-audio", {
//           method: "POST",
//           body: formData,
//         })
//           .then((response) => {
//             if (!response.ok) {
//               throw new Error("Network response was not OK");
//             }
//             return response.json();
//           })
//           .then((data) => {
//             console.log("Backend response received:", data);
//             if (data.error) {
//               alert(`Error: ${data.error}`);
//             } else {
//               setPredictedEmotion(data.emotion); // Update the state with the predicted emotion
//             }
//           })
//           .catch((error) => console.error("Error uploading audio:", error));
//       };
  
//       // Start recording
//       mediaRecorder.start();
//       setIsRecording(true);
//       console.log("Recording started.");
//     } catch (error) {
//       console.error("Error accessing microphone:", error);
//     }
//   };
  
  

//   const stopRecording = () => {
//     const mediaRecorder = mediaRecorderRef.current;

//     if (mediaRecorder && mediaRecorder.state === "recording") {
//       mediaRecorder.stop(); // Stop recording
//       setIsRecording(false);
//       console.log("Recording stopped.");
//     } else if (!mediaRecorder) {
//       console.error("MediaRecorder is not initialized.");
//     } else {
//       console.error(`MediaRecorder is in "${mediaRecorder.state}" state and cannot stop.`);
//     }
//   };

//   return (
//     <section className="speaker-card">
//       <h1>Sample our voice analysis tool!</h1>
//       <div className="recording-controls">
//       <button onClick={startRecording} disabled={isRecording}>
//         🎤 Start Recording
//       </button>
//       <button onClick={stopRecording} disabled={!isRecording}>
//         ⏹ Stop Recording
//       </button>
//       {audioSrc && (
//         <audio src={audioSrc} controls></audio>
//       )}
//       </div>
//       <span className="emotion2">Predicted emotion is: <span className="emotion">{predictedEmotion}</span></span>
//     </section>
//   );
// };

// export default Speaker;

import React, { useState, useRef } from "react";

const Speaker = ({ setVoiceAnalysisResult }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [audioSrc, setAudioSrc] = useState(null);
  const [predictedEmotion, setPredictedEmotion] = useState(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const options = { mimeType: "audio/webm" };
      const mediaRecorder = new MediaRecorder(stream, options);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];
      mediaRecorder.ondataavailable = (event) => audioChunksRef.current.push(event.data);
      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: "audio/webm" });
        const audioUrl = URL.createObjectURL(audioBlob);
        setAudioSrc(audioUrl);

        const formData = new FormData();
        formData.append("audio", audioBlob, "recording.webm");
        const response = await fetch("http://localhost:5001/upload-audio", {
          method: "POST",
          body: formData,
        });
        const data = await response.json();
        if (!data.error) {
          setPredictedEmotion(data.emotion);
          const timestamp = new Date().toLocaleString();
          setVoiceAnalysisResult({ emotion: data.emotion, timestamp });
        }
      };
      mediaRecorder.start();
      setIsRecording(true);
    } catch (error) {
      console.error("Error accessing microphone:", error);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  return (
    <section className="speaker-card">
      <h1>Sample our voice analysis tool!</h1>
      <div className="recording-controls">
        <button onClick={startRecording} disabled={isRecording}>
          🎤 Start Recording
        </button>
        <button onClick={stopRecording} disabled={!isRecording}>
          ⏹ Stop Recording
        </button>
        {audioSrc && <audio src={audioSrc} controls></audio>}
      </div>
      <p className="emotion">Predicted emotion: {predictedEmotion || "Not available"}</p>
    </section>
  );
};

export default Speaker;
