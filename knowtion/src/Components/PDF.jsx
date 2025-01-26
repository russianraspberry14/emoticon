import React, { useState } from 'react';
import jsPDF from 'jspdf';

const PDF = ({ emotions }) => {
    const generatePDF = () => {
        const doc = new jsPDF();

        // Title
        doc.setFontSize(16);
        doc.text('Emotion Detection Report', 10, 10);

        // Emotion Log
        doc.setFontSize(12);
        emotions.forEach((emotion, index) => {
            doc.text(`Time: ${emotion.timestamp} - Emotion: ${emotion.emotion}`, 10, 20 + (index * 10));
        });

        // Download the PDF
        doc.save('emotion_report.pdf');
    };

    return (
        <div className="report-card">
            <h2>Emotion Report</h2>
            <button onClick={generatePDF}>Generate PDF</button>
        </div>
    );
};

export default PDF;