from flask import Flask, jsonify, request, render_template, redirect, url_for
import csv
import os

app = Flask(__name__)

CSV_FILE = 'incidents.csv'

# Helper function to read incidents from the CSV file
def read_incidents():
    with open(CSV_FILE, mode='r') as file:
        reader = csv.DictReader(file)
        return list(reader)

# Helper function to write incidents to the CSV file
def write_incident(incident):
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=incident.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(incident)

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

@app.route('/api/create', methods=['POST'])
def create_incident():
    data = request.json
    if not data:
        return jsonify({'error': 'Invalid data'}), 400
    write_incident(data)
    return jsonify({'message': 'Incident created successfully'}), 201


if __name__ == '__main__':
    app.run(debug=True)