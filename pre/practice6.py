import psycopg2

conn = psycopg2.connect(
    dbname="testpy",
    user="postgres",
    password="1234",
    host="localhost",
    port="5432"
)
cur = conn.cursor()

# Create table
# cur.execute("""
#     CREATE TABLE IF NOT EXISTS students (
#         id SERIAL PRIMARY KEY,
#         name VARCHAR(50),
#         age INT,
#         marks FLOAT
#     );
# """)

# # Insert sample data
# cur.execute("INSERT INTO students (name, age, marks) VALUES (%s, %s, %s)", ("John", 18, 87.5))
# cur.execute("INSERT INTO students (name, age, marks) VALUES (%s, %s, %s)", ("Asha", 19, 92.0))
# cur.execute("INSERT INTO students (name, age, marks) VALUES (%s, %s, %s)", ("Maya", 18, 76.3))

# conn.commit()
# print("Sample data inserted.")

cur.execute("SELECT * FROM students;")
rows = cur.fetchall()

for row in rows:
    print("ID:", row[0], "| Name:", row[1], "| Age:", row[2], "| Marks:", row[3])

cur.close()
conn.close()
