import React, { useState, useEffect } from 'react';

const IndividualAppointmentReminder = ({ appointment, onClose, position }) => {
    return (
        <div style={{
            position: 'fixed',
            top: `${80 + (position * 220)}px`,
            right: '20px',
            backgroundColor: '#f0f0f0',
            padding: '15px',
            borderRadius: '8px',
            boxShadow: '0 2px 5px rgba(0,0,0,0.2)',
            zIndex: 1000 + position,
            maxWidth: '300px'
        }}>
            <h3 style={{ marginTop: 0, marginBottom: '10px', color: '#333' }}>
                Upcoming Appointment
            </h3>
            <p style={{ margin: '5px 0' }}>Dr. {appointment.doctor}</p>
            <p style={{ margin: '5px 0' }}>Date: {appointment.date}</p>
            <p style={{ margin: '5px 0' }}>Email: {appointment.email}</p>
            <button
                onClick={onClose}
                style={{
                    marginTop: '10px',
                    padding: '5px 10px',
                    backgroundColor: '#007bff',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px'
                }}
            >
                Dismiss
            </button>
        </div>
    );
};

const AppointmentReminder = ({ appointments = [] }) => {
    const [activeReminders, setActiveReminders] = useState([]);

    useEffect(() => {
        const checkAppointmentReminders = () => {
            const today = new Date();
            const todayStr = today.toISOString().split('T')[0];
            const newActiveReminders = appointments.filter(appointment => {
                if (!appointment) return false;

                const appointmentDate = new Date(appointment.date);
                const twoDaysBefore = new Date(appointmentDate);
                twoDaysBefore.setDate(appointmentDate.getDate() - 2);

                return (
                    appointment.date === todayStr || // Include today's date
                    (today >= twoDaysBefore && today < appointmentDate)
                );
            });

            setActiveReminders(newActiveReminders);
        };

        checkAppointmentReminders();
        const intervalId = setInterval(checkAppointmentReminders, 3600000);

        return () => clearInterval(intervalId);
    }, [appointments]);

    const dismissReminder = (appointmentId) => {
        setActiveReminders(prev =>
            prev.filter(appointment => appointment.id !== appointmentId)
        );
    };

    return (
        <>
            {activeReminders.map((appointment, index) => (
                <IndividualAppointmentReminder
                    key={appointment.id}
                    appointment={appointment}
                    onClose={() => dismissReminder(appointment.id)}
                    position={index}
                />
            ))}
        </>
    );
};

export default AppointmentReminder;