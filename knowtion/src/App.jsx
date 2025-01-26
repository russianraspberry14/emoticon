import React from 'react';
import Navbar from './Components/Navbar';
import About from './Components/About';
import Appointments from './Components/Appointments';
import PDF from './Components/PDF';
import Speaker from './Components/Speaker';
import Simulator from './Components/Simulator'
import './index.css';

function App() {
    return (
        <div>
            <Navbar />
            <div style={{ marginTop: '80px', padding: '20px' }}>
                <h1>Welcome to your Dashboard, Ekansh</h1>
                <p>Build your knowledge with ease!</p>
                <div className="grid-container">
                  <About />
                  <Appointments />
                  <PDF />
                  <Speaker />
                  <Simulator />
                </div>
            </div>
        </div>
    );
}

export default App;
