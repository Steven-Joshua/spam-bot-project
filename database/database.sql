CREATE DATABASE spam_detection_db;

USE spam_detection_db;

CREATE TABLE message_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source VARCHAR(10),
    message_text TEXT,
    spam_probability FLOAT,
    prediction VARCHAR(10),
    file_path VARCHAR(255)
);
ALTER TABLE message_logs MODIFY COLUMN source VARCHAR(255);
DESC message_logs;
ALTER TABLE message_logs MODIFY COLUMN source VARCHAR(500);
ALTER TABLE message_logs MODIFY COLUMN spam_probability FLOAT;
ALTER TABLE message_logs DROP COLUMN file_path;



