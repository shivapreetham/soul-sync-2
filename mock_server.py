```python
from flask import Flask, request, jsonify
import os
from google.generativeai import GenerativeAI
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Set API key from environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in .env file")
gen_ai = GenerativeAI(api_key=GEMINI_API_KEY)

@app.route('/gemini', methods=['POST'])
def gemini():
    try:
        data = request.get_json()
        message = data.get('message')
        if not message:
            return jsonify({"error": "Message is required"}), 400
        model = gen_ai.GenerativeModel('gemini-1.5-flash')
        result = model.generate_content(message)
        text = result.text
        return jsonify({"response": text})
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": "Failed to process Gemini request"}), 500

if __name__ == '__main__':
    app.run(port=5000)
```