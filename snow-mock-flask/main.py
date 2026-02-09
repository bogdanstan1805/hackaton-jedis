from flask import Flask, jsonify, request, render_template, redirect, url_for
import csv
import requests

app = Flask(__name__)

CSV_FILE = 'incidents.csv'

# Helper function to read incidents from the CSV file
def read_incidents():
    with open(CSV_FILE, mode='r') as file:
        reader = csv.DictReader(file)
        return list(reader)

# Helper function to write incidents to the CSV file
# def write_incident(incident):
#     file_exists = os.path.isfile(CSV_FILE)
#     with open(CSV_FILE, mode='a', newline='') as file:
#         writer = csv.DictWriter(file, fieldnames=incident.keys())
#         if not file_exists:
#             writer.writeheader()
#         writer.writerow(incident)

@app.route('/')
def index():
    incidents = read_incidents()
    return render_template('index.html', incidents=incidents)

@app.route('/incident/<string:number>')
def view_incident(number):
    incidents = read_incidents()
    incident = next((i for i in incidents if i['number'] == number), None)
    if not incident:
        return "Incident not found", 404
    return render_template('incident.html', incident=incident)

@app.route('/api/getAll', methods=['GET'])
def get_all():
    incidents = read_incidents()
    return jsonify(incidents)

@app.route('/chat', methods=['POST', 'GET'])
def chat():

    if request.method == 'GET':
        # Render the chat page without any messages
        return render_template('chat.html')

    print (request.form)

    data = request.get_json()
    user_input = data.get('message')

    print(user_input)
    if not user_input:
        return "No input provided", 400

    try:
        # Forward the input to the external service
        response = requests.post('http://127.0.0.1:5001/chat', json={'input': user_input})
        response.raise_for_status()  # Raise an error for HTTP errors
        assistant_response = response.json().get('response', 'No response from assistant')

        # Render the chat page with the user input and assistant response
        return render_template('chat.html', userMessage=user_input, assistantMessage=assistant_response)
    except requests.RequestException as e:
        return f"Error communicating with the chat service: {e}", 500


# @app.route('/api/create', methods=['POST'])
# def create_incident():
#     data = request.json
#     if not data:
#         return jsonify({'error': 'Invalid data'}), 400
#     write_incident(data)
#     return jsonify({'message': 'Incident created successfully'}), 201


if __name__ == '__main__':
    app.run(debug=True)