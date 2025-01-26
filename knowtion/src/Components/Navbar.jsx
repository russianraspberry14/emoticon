import React, { useState, useEffect } from 'react';
import image from '../assets/1.png';
import './navbar.css';

const Navbar = () => {
    const [showNavbar, setShowNavbar] = useState(true);
    const [lastScrollY, setLastScrollY] = useState(0);

    const handleScroll = () => {
        const currentScrollY = window.scrollY;

        if (currentScrollY > lastScrollY && currentScrollY > 50) {
            // Scrolling down, hide the navbar
            setShowNavbar(false);
        } else {
            // Scrolling up, show the navbar
            setShowNavbar(true);
        }

        setLastScrollY(currentScrollY);
    };

    useEffect(() => {
        window.addEventListener('scroll', handleScroll);
        return () => {
            window.removeEventListener('scroll', handleScroll);
        };
    }, [lastScrollY]);

    return (
        <nav className={`navbar ${showNavbar ? 'visible' : 'hidden'}`}>
            <div className="brand">
                <img src={image} alt="Knowtion" className="navbar-image" />
            </div>
        </nav>
    );
};

export default Navbar;
