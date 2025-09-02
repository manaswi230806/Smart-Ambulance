
from flask import Flask, render_template, request, jsonify
import json
import openrouteservice
from openrouteservice import convert
from geopy.distance import geodesic
from datetime import datetime

app = Flask(__name__)

ORS_API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6ImVlYzU2MTQ5NDA4ODQ5NjJiYWYwNzE2ZTEwYTEwMjA5IiwiaCI6Im11cm11cjY0In0="
client = openrouteservice.Client(key=ORS_API_KEY)

with open("hospitals.json", "r") as f:
    hospitals = json.load(f)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/route', methods=['POST'])
def get_route():
    data = request.get_json()
    user_lat = data['latitude']
    user_lng = data['longitude']
    patient = data['patient']

    user_loc = (user_lat, user_lng)
    nearest_hospital = None
    shortest_distance = float('inf')

    for hospital in hospitals:
        hosp_loc = (hospital["lat"], hospital["lng"])
        dist = geodesic(user_loc, hosp_loc).km
        if dist < shortest_distance:
            shortest_distance = dist
            nearest_hospital = hospital

    coords = [(user_lng, user_lat), (nearest_hospital["lng"], nearest_hospital["lat"])]
    try:
        route = client.directions(coords, profile='driving-car', format='geojson')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "route": route,
        "hospital": nearest_hospital,
        "patient": patient
    })

if __name__ == '__main__':
    app.run(debug=True)
