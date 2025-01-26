import React from "react";
import jsPDF from "jspdf";

const PDF = ({ lastAppointment, nextAppointment, voiceAnalysisResult }) => {
    const generatePDF = () => {
        const doc = new jsPDF();

        // Add a header with styling
        doc.setFontSize(20);
        doc.setFont("Helvetica", "bold");
        doc.text("Medical Appointment Report", 105, 15, null, null, "center");

        // Add line separator
        doc.setLineWidth(0.5);
        doc.line(10, 20, 200, 20);

        // Section: Last Appointment
        doc.setFontSize(16);
        doc.setFont("Helvetica", "bold");
        doc.text("Last Appointment", 10, 30);
        doc.setLineWidth(0.1);
        doc.line(10, 32, 200, 32);

        if (lastAppointment) {
            doc.setFontSize(12);
            doc.setFont("Helvetica", "normal");
            doc.text(`Doctor: ${lastAppointment.doctor}`, 10, 40);
            doc.text(`Date: ${lastAppointment.date}`, 10, 48);
        } else {
            doc.setFontSize(12);
            doc.setFont("Helvetica", "italic");
            doc.text("No past appointments found.", 10, 40);
        }

        // Section: Voice Analysis Result
        doc.setFontSize(16);
        doc.setFont("Helvetica", "bold");
        doc.text("Voice Analysis Result", 10, 60);
        doc.setLineWidth(0.1);
        doc.line(10, 62, 200, 62);

        if (voiceAnalysisResult) {
            doc.setFontSize(12);
            doc.setFont("Helvetica", "normal");
            doc.text(`Predicted Emotion: ${voiceAnalysisResult.emotion}`, 10, 70);
            doc.text(`Timestamp: ${voiceAnalysisResult.timestamp}`, 10, 78);
        } else {
            doc.setFontSize(12);
            doc.setFont("Helvetica", "italic");
            doc.text("No voice analysis results available.", 10, 70);
        }

        // Section: Next Appointment
        doc.setFontSize(16);
        doc.setFont("Helvetica", "bold");
        doc.text("Next Appointment", 10, 90);
        doc.setLineWidth(0.1);
        doc.line(10, 92, 200, 92);

        if (nextAppointment) {
            doc.setFontSize(12);
            doc.setFont("Helvetica", "normal");
            doc.text(`Doctor: ${nextAppointment.doctor}`, 10, 100);
            doc.text(`Date: ${nextAppointment.date}`, 10, 108);
        } else {
            doc.setFontSize(12);
            doc.setFont("Helvetica", "italic");
            doc.text("No upcoming appointments scheduled.", 10, 100);
        }

        // Add footer with disclaimer
        doc.setFontSize(10);
        doc.setFont("Helvetica", "italic");
        doc.text(
            "This report was generated dynamically and may not reflect real-time updates.",
            105,
            285,
            null,
            null,
            "center"
        );

        // Save the PDF
        doc.save("medical_appointments.pdf");
    };

    return (
        <div className="report-card">
            <h2>Generate Medical Report</h2>
            <button onClick={generatePDF} className="generate-button">
                Generate PDF
            </button>
        </div>
    );
};

export default PDF;
