import psycopg2
from psycopg2 import sql
from config import DATABASE_URL  # Import connection URL from config.py

def get_db_connection():
    """Establishes a database connection with PostgreSQL."""
    try:
        connection = psycopg2.connect(DATABASE_URL, sslmode="require")
        return connection
    except psycopg2.Error as err:
        print(f"Database connection failed: {err}")
        return None

def insert_message(source, message_text, spam_probability, prediction, file_path=None):
    """Inserts a classified message into the PostgreSQL database."""
    connection = get_db_connection()
    if connection is None:
        print("Database connection failed. Message not logged.")
        return
    
    try:
        # Ensure spam_probability is a float
        try:
            spam_probability = float(spam_probability) if spam_probability not in [None, ""] else 0.0
        except ValueError:
            print(f"Invalid spam_probability value: {spam_probability}. Setting to 0.0.")
            spam_probability = 0.0

        # Debugging: Print values before inserting
        print(f"Inserting: source={source}, message_text={message_text}, "
              f"spam_probability={spam_probability}, type={type(spam_probability)}, "
              f"prediction={prediction},")

        cursor = connection.cursor()
        sql_query = """INSERT INTO message_logs 
                       (source, message_text, spam_probability, prediction, timestamp) 
                       VALUES (%s, %s, %s, %s, NOW())"""
        values = (source, message_text, spam_probability, prediction)

        cursor.execute(sql_query, values)
        connection.commit()
        print("Message logged successfully.")

    except psycopg2.Error as err:
        print(f"Database error: {err}")
    
    finally:
        cursor.close()
        connection.close()

# Example call (test this function)
if __name__ == "__main__":
    insert_message("email", "This is a spam message", "0.75", "Spam")
