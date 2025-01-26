import React, { useState } from "react";
import AppointmentReminder from './AppointmentReminder';

const Appointments = () => {
    const [appointments, setAppointments] = useState([
        { id: 1, doctor: "Dr. John Doe", date: "2025-02-01", email: "john.doe@example.com" },
        { id: 2, doctor: "Dr. Jane Smith", date: "2025-02-15", email: "jane.smith@example.com" },
    ]);

    const [newDoctor, setNewDoctor] = useState("");
    const [newDate, setNewDate] = useState("");
    const [newEmail, setNewEmail] = useState("");

    const addAppointment = () => {
        if (!newDoctor || !newDate || !newEmail) {
            alert("Please fill out all fields.");
            return;
        }
        const newAppointment = {
            id: Date.now(),
            doctor: newDoctor,
            date: newDate,
            email: newEmail,
        };
        setAppointments([...appointments, newAppointment]);
        setNewDoctor("");
        setNewDate("");
        setNewEmail("");
    };

    const deleteAppointment = (id) => {
        const updatedAppointments = appointments.filter((appointment) => appointment.id !== id);
        setAppointments(updatedAppointments);
    };

    return (
        <section>
            <h2>Appointments</h2>
            <AppointmentReminder appointments={appointments} />

            {/* Rest of the existing component remains the same */}
            <div className="appointments-list">
                {appointments.map((appointment) => (
                    <div key={appointment.id} className="appointment-card">
                        <h3>{appointment.doctor}</h3>
                        <p>Date: {appointment.date}</p>
                        <p>Email: {appointment.email}</p>
                        <button
                            className="contact-button"
                            onClick={() => (window.location = `mailto:${appointment.email}`)}
                        >
                            Contact
                        </button>
                        <button
                            onClick={() => deleteAppointment(appointment.id)}
                            className="delete-button"
                        >
                            Delete
                        </button>
                    </div>
                ))}
            </div>

            {/* Add New Appointment form remains the same */}
            <div className="add-appointment">
                <h3>Create New Appointment</h3>
                <input
                    type="text"
                    placeholder="Doctor's Name"
                    value={newDoctor}
                    onChange={(e) => setNewDoctor(e.target.value)}
                />
                <input
                    type="date"
                    value={newDate}
                    onChange={(e) => setNewDate(e.target.value)}
                />
                <input
                    type="email"
                    placeholder="Doctor's Email"
                    value={newEmail}
                    onChange={(e) => setNewEmail(e.target.value)}
                />
                <button onClick={addAppointment} className="add-button">
                    Add Appointment
                </button>
            </div>
        </section>
    );
};

export default Appointments;