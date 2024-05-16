import sqlite3

# Polaczenie do bazy danych
conn = sqlite3.connect('crypto_predictions.db')

# Zapytanie do wyświetlenia wszystkich danych z tabeli predictions
query_all = "SELECT * FROM predictions"
cursor_all = conn.execute(query_all)
for row in cursor_all:
    print(row)

# # Zapytanie do wyświetlenia daty, kryptowaluty i predykcji dla rekordów z wartością predykcji powyżej 1000
# query_above_1000 = "SELECT date, crypto, prediction FROM predictions WHERE prediction > 1000"
# cursor_above_1000 = conn.execute(query_above_1000)
# for row in cursor_above_1000:
#     print(row)
#
# # Zapytanie do wyświetlenia unikalnych kryptowalut w tabeli predictions
# query_unique_crypto = "SELECT DISTINCT crypto FROM predictions"
# cursor_unique_crypto = conn.execute(query_unique_crypto)
# for row in cursor_unique_crypto:
#     print(row)
#
# # Zapytanie do wyświetlenia średniej wartości predykcji dla danej kryptowaluty
# query_avg_prediction = "SELECT crypto, AVG(prediction) AS avg_prediction FROM predictions GROUP BY crypto"
# cursor_avg_prediction = conn.execute(query_avg_prediction)
# for row in cursor_avg_prediction:
#     print(row)

# Zamknięcie połączenia z bazą danych
conn.close()
