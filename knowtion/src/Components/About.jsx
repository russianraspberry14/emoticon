import React, { useState } from "react";

const About = () => {
    // Array of card data
    const cards = [
        { id: 1, title: "Heart Rate", description: "We utilize your heartbeat and motion to analyze your stressful moments to keep track when you can't!" },
        { id: 2, title: "Voice Analysis", description: "Detect your tone while having conversations to analyze for any mood swings or extreme moods" },
        { id: 3, title: "Comprehensive Reports", description: "Generate comprehensive reports" },
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
