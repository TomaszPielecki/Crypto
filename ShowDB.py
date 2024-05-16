import sqlite3

# Establishing a connection to the database
conn = sqlite3.connect('crypto_predictions.db')

# Query to retrieve all data from the predictions table
query_all = "SELECT * FROM predictions"
cursor_all = conn.execute(query_all)
for row in cursor_all:
    print(row)
# Closing the connection to the database
conn.close()
