import React from 'react';
import image from '../assets/1.png';
import './navbar.css';

const Navbar = () => {
    return (
        <nav className="navbar">
            <div className="brand">
                <img src={image} alt="Knowtion" />
            </div>
            <ul className="links">
                <li><a href="#">Home</a></li>
                <li><a href="#">Link 2</a></li>
                <li><a href="#">Link 3</a></li>
                <li className="dropdown">
                    <a href="#">Dropdown</a>
                    <div className="dropdown-content">
                        <a href="#">Dropdown Item 1</a>
                        <a href="#">Dropdown Item 2</a>
                        <a href="#">Dropdown Item 3</a>
                    </div>
                </li>
            </ul>
        </nav>
    );
};

export default Navbar;
