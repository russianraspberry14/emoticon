const express = require("express");
const multer = require("multer");
const { PythonShell } = require("python-shell");
const cors = require("cors");
const path = require("path");
const { exec } = require("child_process");
const app = express();

// Configure CORS
app.use(
  cors({
    origin: "http://localhost:5173", // Allow requests from your React frontend
    methods: ["GET", "POST"],       // Allow GET and POST requests
    allowedHeaders: ["Content-Type"], // Allow headers
  })
);

// Configure multer to handle file uploads
const upload = multer({ dest: "uploads/" });

// Define the upload route
app.post("/upload-audio", upload.single("audio"), (req, res) => {
    console.log("Received a request to /upload-audio");
    console.log("Python script absolute path:", path.resolve(__dirname, "process_audio.py"));

    const inputPath = path.resolve(req.file.path); // Path to the uploaded WebM file
    const outputPath = `${inputPath}.wav`; // Path for the converted WAV file
  
    // Convert WebM to WAV using ffmpeg
    exec(`ffmpeg -i ${inputPath} -ar 16000 -ac 1 ${outputPath}`, (err) => {
      if (err) {
        console.error("Error converting WebM to WAV:", err);
        return res.status(500).json({ error: "Error converting audio file." });
      }
  
      // Process the converted WAV file with the Python script
      console.log("Calling Python script with file:", req.file.path);
      PythonShell.run(
        "process_audio.py",
        {
            args: [outputPath],
            pythonPath: "/Users/ekanshsahu/Documents/emoticon/backend/venv/bin/python3",
            pythonOptions: ["-u"],
        },
        (err, result) => {
            console.log("Raw Result:", result);
            console.log("Error Object:", err);
            console.log("Full Error Details:", err?.message, err?.stack);
            
            if (err) {
                return res.status(500).json({ error: "Python script error", details: err.message });
            }
    
            if (!result || result.length === 0) {
                console.error("No output from Python script");
                return res.status(500).json({ error: "No output" });
            }
    
            const predictedEmotion = result[0]?.trim() || "Unknown";
            res.json({ emotion: predictedEmotion });
        }
    );
    
    });
  });

// Start the server
const PORT = 5001;
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
