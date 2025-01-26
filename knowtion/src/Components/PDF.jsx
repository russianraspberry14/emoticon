import React from 'react';
import jsPDF from 'jspdf';

const PDF = () => {
    const generatePDF = () => {
        const doc = new jsPDF();

        // Add content to the PDF
        doc.text('Knowtion PDF Example', 10, 10); // (text, x, y)
        doc.text('This is a dynamically generated PDF!', 10, 20);
        doc.text('Enjoy your PDF creation journey.', 10, 30);

        // Download the PDF
        doc.save('example.pdf');
    };

    return (
        <div className="report-card">
            <h2>Report Generator</h2>
            <button onClick={generatePDF}>Generate PDF</button>
        </div>
    );
};

export default PDF;