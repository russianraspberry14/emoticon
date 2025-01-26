import React from 'react';
import Navbar from './Components/Navbar';
import About from './Components/About';
import Appointments from './Components/Appointments';
import Speaker from './Components/Speaker';
import './index.css';

function App() {
    return (
        <div>
            <Navbar />
            <div style={{ marginTop: '80px', padding: '20px' }}>
                <h1>Welcome to your Dashboard, User!</h1>
                <p>Keep track of your mood and appointments!</p>
                <div className="grid-container">
                  <About />
                  <Appointments />
                  <Speaker />
                </div>
            </div>
        </div>
    );
}

export default App;
