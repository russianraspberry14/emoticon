import React, { useState } from "react";

const About = () => {
    // Array of card data
    const cards = [
        {
            id: 1, title: "Heart Rate", description: "We are in sync with your heartbeat."
                + "\nOur heart tracker will help you record aberrations your heartbeat to analyze your stressful moments and keep track when you can't!"
        },
        {
            id: 2, title: "Voice Analysis", description: "Our state-of-the-art emotion detection software is keeping an ear out for you."
                + "\nWe listen to your daily interactions and note sudden changes in tone, so that you keep track and introspect on your day later."
        },
        {
            id: 3, title: "Comprehensive Reports", description: "From the data you provide us, we generate reports."
                + "\nWe log incidents that stand out, along with their timings."
                + "\nWe help you conceptualize just how much your interactions affect your mental state over the course of a typical day."
        },
        {
            id: 4, title: "Medical Collaboration", description: "Manage and track your medical appointments."
                + "\nNever miss an appointment again, as we start reminding you two days in advance."
                + "\nYour approved doctors can view your in-app reports."
                + "\nEase of communication between different healthcare providers through our app reduces medical miscommunication."
        },
    ];

    // State to track the current card index
    const [currentCardIndex, setCurrentCardIndex] = useState(0);

    // Handle "Next" button click
    const handleNext = () => {
        setCurrentCardIndex((prevIndex) => (prevIndex + 1) % cards.length);
    };

    return (
        <section>
            <h2>About</h2>
            <div className="card">
                <h3>{cards[currentCardIndex].title}</h3>
                <p>{cards[currentCardIndex].description}</p>
            </div>
            <button onClick={handleNext} className="next-button">
                Next
            </button>
        </section>
    );
};

export default About;
