from flask import Flask, request, jsonify
from flask_cors import cors
import pymysql

app = Flask(__name__)
cors(app)  # Allow browser-based JS to access this API

def get_connection():
    return pymysql.connect(
        host="localhost",
        user="root",
        password="",
        database="your_db_name"
    )

@app.route("/chat", methods=["POST"])
def chatbot_reply():
    user_msg = request.json.get("message", "")
    # Example: save message or fetch from DB
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO chats (message) VALUES (%s)", (user_msg,))
    conn.commit()
    conn.close()

    # Dummy logic
    response = "I got your message: " + user_msg
    return jsonify({"reply": response})

if __name__ == "__main__":
    app.run(debug=True)
