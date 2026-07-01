import sqlite3


def build_dummy_database():
    database_name = "company_store.db"
    print(f"🛠️ Creating local commercial database: {database_name}...")

    # Connect to SQLite
    conn = sqlite3.connect(database_name)
    cursor = conn.cursor()

    # 1. Drop old tables if they exist to keep data fresh
    cursor.execute("DROP TABLE IF EXISTS AggregatedSpendReporting;")

    # 2. Create the unified Aggregated Spend Reporting Table matching your schema
    cursor.execute("""
    CREATE TABLE AggregatedSpendReporting (
        RecordID INTEGER PRIMARY KEY AUTOINCREMENT,
        FrameworkNumber TEXT NOT NULL,
        SupplierName TEXT NOT NULL,
        FinancialYear TEXT NOT NULL,
        EvidencedSpend REAL NOT NULL
    );
    """)

    # 3. Insert Dummy Data based on your screenshot and requested supplier factsheets
    # Replicating multi-million pound spends across UK public sector frameworks
    mock_spend_records = [
        # Accenture Records (Matching your image data profile)
        ('RM6100', 'ACCENTURE (UK) LIMITED', '2025/26', 147571692.69),
        ('RM6335', 'ACCENTURE (UK) LIMITED', '2025/26', 79922465.54),
        ('RM6263', 'ACCENTURE (UK) LIMITED', '2025/26', 32839602.27),
        ('RM6221', 'ACCENTURE (UK) LIMITED', '2024/25', 30377111.26),
        ('RM1043.8', 'ACCENTURE (UK) LIMITED', '2024/25', 18128255.68),

        # BAE Systems Records
        ('RM6187', 'BAE SYSTEMS APPLIED INTELLIGENCE LIMITED', '2025/26', 92450000.00),
        ('RM3804', 'BAE SYSTEMS APPLIED INTELLIGENCE LIMITED', '2025/26', 15966478.93),
        ('RM6100', 'BAE SYSTEMS APPLIED INTELLIGENCE LIMITED', '2024/25', 45120000.50),

        # Microsoft Records
        ('RM6100', 'MICROSOFT IRELAND OPERATIONS LIMITED', '2025/26', 185300000.00),
        ('RM1557.13', 'MICROSOFT IRELAND OPERATIONS LIMITED', '2025/26', 12697445.16),
        ('RM3764.3', 'MICROSOFT IRELAND OPERATIONS LIMITED', '2024/25', 12450066.20),
        ('RM6193', 'MICROSOFT IRELAND OPERATIONS LIMITED', '2024/25', 11779350.95)
    ]

    cursor.executemany("""
        INSERT INTO AggregatedSpendReporting (FrameworkNumber, SupplierName, FinancialYear, EvidencedSpend) 
        VALUES (?, ?, ?, ?);
    """, mock_spend_records)

    # Save changes and close connection
    conn.commit()
    conn.close()
    print("✅ Local mock database built successfully with [AggregatedSpendReporting] records!")


# if __name__ == "__main__":
#     build_dummy_database()