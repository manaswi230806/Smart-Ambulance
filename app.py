# from flask import Flask, render_template, request, jsonify, redirect, url_for
# import json
# import os
# import openrouteservice
# from openrouteservice import convert
# from geopy.distance import geodesic
# from datetime import datetime
# import time
# import uuid

# app = Flask(__name__)

# ORS_API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6ImVlYzU2MTQ5NDA4ODQ5NjJiYWYwNzE2ZTEwYTEwMjA5IiwiaCI6Im11cm11cjY0In0="
# client = openrouteservice.Client(key=ORS_API_KEY)

# pending_requests = []
# accepted_trips = {}

# REPORTS_FOLDER = 'patient_reports'
# if not os.path.exists(REPORTS_FOLDER):
#     os.makedirs(REPORTS_FOLDER)

# with open("hospitals.json", "r") as f:
#     hospitals = json.load(f)

# medchal_hospitals = [
#     "Leela Multispeciality – Medchal",
#     "VVR Hospital – Medchal",
#     "Dr Raju’s Hospital – Medchal",
#     "Medicity Hospital – Medchal",
#     "Pavithran Hospital – Medchal",
#     "Sankhya Hospitals – Medchal"
# ]

# @app.route('/')
# def home():
#     return render_template("index.html")

# @app.route('/driver')
# def driver_dashboard():
#     return render_template("driver_dashboard.html")

# @app.route('/status/<request_id>')
# def status_page(request_id):
#     return render_template("status.html", request_id=request_id)

# @app.route('/hospitals')
# def get_hospitals():
#     return jsonify(hospitals)

# @app.route('/get_requests')
# def get_requests():
#     """Endpoint for drivers to get pending requests."""
#     # Check if a driver is currently busy with an accepted trip
#     is_driver_busy = False
#     for trip_id in accepted_trips:
#         if accepted_trips[trip_id]['status'] not in ['arrived_at_hospital', 'completed']:
#             is_driver_busy = True
#             break
            
#     if is_driver_busy:
#         return jsonify([])
#     else:
#         return jsonify(pending_requests)

# @app.route('/update_trip_status', methods=['POST'])
# def update_trip_status():
#     data = request.get_json()
#     request_id = data.get('request_id')
#     new_status = data.get('status')
    
#     trip = accepted_trips.get(request_id)
#     if trip:
#         trip['status'] = new_status
#         if new_status == 'arrived_at_hospital':
#             trip['dropoff_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#             del accepted_trips[request_id]
#             print(f"Trip {request_id} completed.")
#         return jsonify({"message": f"Trip status updated to {new_status}."})
    
#     return jsonify({"error": "Trip not found."}), 404

# @app.route('/get_trip_status/<request_id>')
# def get_trip_status(request_id):
#     trip = accepted_trips.get(request_id)
#     if trip:
#         return jsonify({
#             'status': trip.get('status', 'enroute_to_patient'),
#             'patient_location': trip.get('user_location'),
#             'destination_hospital': trip.get('destination_hospital'),
#             'patient': trip.get('patient')
#         })
#     return jsonify({"status": "pending", "message": "Awaiting ambulance dispatch."})

# @app.route('/accept_trip', methods=['POST'])
# def accept_trip():
#     data = request.get_json()
#     request_id = data.get('request_id')

#     request_data = None
#     for req in pending_requests:
#         if req['request_id'] == request_id:
#             request_data = req
#             break
    
#     if request_data:
#         pending_requests.remove(request_data)
        
#         user_loc = (request_data['user_location']['lat'], request_data['user_location']['lng'])
#         hosp_loc = (request_data['destination_hospital']['lat'], request_data['destination_hospital']['lng'])
#         coords = [(user_loc[1], user_loc[0]), (hosp_loc[1], hosp_loc[0])]
        
#         try:
#             route = client.directions(coords, profile='driving-car', format='geojson')
#         except Exception as e:
#             print(f"Error getting route for accepted trip: {e}")
#             route = None

#         request_data['status'] = 'enroute_to_patient'
#         request_data['pickup_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#         request_data['route_data'] = route
#         accepted_trips[request_id] = request_data
#         print(f"Driver accepted trip {request_id}")
#         return jsonify({"message": "Trip accepted!", "trip": request_data})
    
#     return jsonify({"error": "Request not found."}), 404

# @app.route('/route', methods=['POST'])
# def get_route():
#     user_lat = float(request.form['latitude'])
#     user_lng = float(request.form['longitude'])
#     patient_data = json.loads(request.form['patient'])
#     selected_hospital_name = request.form.get('hospitalName')
#     report_file = request.files.get('report')

#     user_loc = (user_lat, user_lng)
    
#     destination_hospital = None
#     if selected_hospital_name:
#         for hospital in hospitals:
#             if hospital["name"] == selected_hospital_name:
#                 destination_hospital = hospital
#                 break
#     else:
#         shortest_distance = float('inf')
#         for hospital in hospitals:
#             hosp_loc = (hospital["lat"], hospital["lng"])
#             dist = geodesic(user_loc, hosp_loc).km
#             if dist < shortest_distance:
#                 shortest_distance = dist
#                 destination_hospital = hospital
    
#     if not destination_hospital:
#         return jsonify({"error": "No valid hospital found."}), 404

#     report_path = None
#     if report_file:
#         file_extension = os.path.splitext(report_file.filename)[1]
#         unique_filename = f"{patient_data.get('name', 'Unknown')}_{uuid.uuid4()}{file_extension}"
#         file_path = os.path.join(REPORTS_FOLDER, unique_filename)
#         report_file.save(file_path)
#         report_path = file_path
    
#     request_id = str(uuid.uuid4())
    
#     pending_requests.append({
#         'request_id': request_id,
#         'patient': patient_data,
#         'user_location': {'lat': user_lat, 'lng': user_lng},
#         'destination_hospital': destination_hospital,
#         'report_path': report_path,
#         'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     })
    
#     print(f"New request added with ID: {request_id}")
#     return jsonify({"message": "Request received and an ambulance is being dispatched.", "request_id": request_id})

# @app.route('/get_route_details', methods=['POST'])
# def get_route_details():
#     user_lat = float(request.form['latitude'])
#     user_lng = float(request.form['longitude'])
#     patient_data = json.loads(request.form['patient'])
#     selected_hospital_name = request.form.get('hospitalName')
    
#     user_loc = (user_lat, user_lng)
    
#     destination_hospital = None
#     if selected_hospital_name:
#         for hospital in hospitals:
#             if hospital["name"] == selected_hospital_name:
#                 destination_hospital = hospital
#                 break
#     else:
#         shortest_distance = float('inf')
#         for hospital in hospitals:
#             hosp_loc = (hospital["lat"], hospital["lng"])
#             dist = geodesic(user_loc, hosp_loc).km
#             if dist < shortest_distance:
#                 shortest_distance = dist
#                 destination_hospital = hospital
    
#     if not destination_hospital:
#         return jsonify({"error": "No valid hospital found."}), 404

#     # New logic for Medchal hospitals
#     if "Medchal" in destination_hospital["name"]:
#         return jsonify({
#             "error": "Busier",
#             "hospital": destination_hospital,
#             "patient": patient_data,
#             "latitude": user_lat,
#             "longitude": user_lng
#         })

#     coords = [(user_lng, user_lat), (destination_hospital["lng"], destination_hospital["lat"])]
#     try:
#         route = client.directions(coords, profile='driving-car', format='geojson')
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

#     return jsonify({
#         "route": route,
#         "hospital": destination_hospital,
#         "patient": patient_data,
#         "latitude": user_lat,
#         "longitude": user_lng
#     })

# if __name__ == '__main__':
#     app.run(debug=True)




# from flask import Flask, render_template, request, jsonify, redirect, url_for
# import json
# import os
# import openrouteservice
# from openrouteservice import convert
# from geopy.distance import geodesic
# from datetime import datetime
# import time
# import uuid

# app = Flask(__name__)

# ORS_API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6ImVlYzU2MTQ5NDA4ODQ5NjJiYWYwNzE2ZTEwYTEwMjA5IiwiaCI6Im11cm11cjY0In0="
# client = openrouteservice.Client(key=ORS_API_KEY)

# pending_requests = []
# accepted_trips = {}

# REPORTS_FOLDER = 'patient_reports'
# if not os.path.exists(REPORTS_FOLDER):
#     os.makedirs(REPORTS_FOLDER)

# with open("hospitals.json", "r") as f:
#     hospitals = json.load(f)

# medchal_hospitals = [
#     "Leela Multispeciality – Medchal",
#     "VVR Hospital – Medchal",
#     "Dr Raju’s Hospital – Medchal",
#     "Medicity Hospital – Medchal",
#     "Pavithran Hospital – Medchal",
#     "Sankhya Hospitals – Medchal"
# ]

# @app.route('/')
# def home():
#     return render_template("index.html")

# @app.route('/driver')
# def driver_dashboard():
#     return render_template("driver_dashboard.html")

# @app.route('/status/<request_id>')
# def status_page(request_id):
#     return render_template("status.html", request_id=request_id)

# @app.route('/hospitals')
# def get_hospitals():
#     return jsonify(hospitals)

# @app.route('/get_requests')
# def get_requests():
#     is_driver_busy = len(accepted_trips) > 0 and 'arrived_at_hospital' not in [trip['status'] for trip in accepted_trips.values()]
    
#     if is_driver_busy:
#         return jsonify([])
#     else:
#         return jsonify(pending_requests)

# @app.route('/update_trip_status', methods=['POST'])
# def update_trip_status():
#     data = request.get_json()
#     request_id = data.get('request_id')
#     new_status = data.get('status')
    
#     trip = accepted_trips.get(request_id)
#     if trip:
#         trip['status'] = new_status
#         if new_status == 'arrived_at_hospital':
#             trip['dropoff_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#             del accepted_trips[request_id]
#             print(f"Trip {request_id} completed.")
#         return jsonify({"message": f"Trip status updated to {new_status}."})
    
#     return jsonify({"error": "Trip not found."}), 404

# @app.route('/get_trip_status/<request_id>')
# def get_trip_status(request_id):
#     trip = accepted_trips.get(request_id)
#     if trip:
#         return jsonify({
#             'status': trip.get('status', 'enroute_to_patient'),
#             'patient_location': trip.get('user_location'),
#             'destination_hospital': trip.get('destination_hospital'),
#             'patient': trip.get('patient')
#         })
#     return jsonify({"status": "pending", "message": "Awaiting ambulance dispatch."})

# @app.route('/accept_trip', methods=['POST'])
# def accept_trip():
#     data = request.get_json()
#     request_id = data.get('request_id')

#     request_data = None
#     for req in pending_requests:
#         if req['request_id'] == request_id:
#             request_data = req
#             break
    
#     if request_data:
#         pending_requests.remove(request_data)
        
#         user_loc = (request_data['user_location']['lat'], request_data['user_location']['lng'])
#         hosp_loc = (request_data['destination_hospital']['lat'], request_data['destination_hospital']['lng'])
#         coords = [(user_loc[1], user_loc[0]), (hosp_loc[1], hosp_loc[0])]
        
#         try:
#             route = client.directions(coords, profile='driving-car', format='geojson')
#         except Exception as e:
#             print(f"Error getting route for accepted trip: {e}")
#             route = None

#         request_data['status'] = 'enroute_to_patient'
#         request_data['pickup_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#         request_data['route_data'] = route
#         accepted_trips[request_id] = request_data
#         print(f"Driver accepted trip {request_id}")
#         return jsonify({"message": "Trip accepted!", "trip": request_data})
    
#     return jsonify({"error": "Request not found."}), 404

# @app.route('/route', methods=['POST'])
# def get_route():
#     user_lat = float(request.form['latitude'])
#     user_lng = float(request.form['longitude'])
#     patient_data = json.loads(request.form['patient'])
#     selected_hospital_name = request.form.get('hospitalName')
#     report_file = request.files.get('report')

#     user_loc = (user_lat, user_lng)
    
#     destination_hospital = None
#     if selected_hospital_name:
#         for hospital in hospitals:
#             if hospital["name"] == selected_hospital_name:
#                 destination_hospital = hospital
#                 break
#     else:
#         shortest_distance = float('inf')
#         for hospital in hospitals:
#             hosp_loc = (hospital["lat"], hospital["lng"])
#             dist = geodesic(user_loc, hosp_loc).km
#             if dist < shortest_distance:
#                 shortest_distance = dist
#                 destination_hospital = hospital
    
#     if not destination_hospital:
#         return jsonify({"error": "No valid hospital found."}), 404

#     # Check for "Busier" condition and return an error. The front-end handles the pop-up.
#     if destination_hospital['name'] in medchal_hospitals:
#         return jsonify({"error": "Busier"})
    
#     report_path = None
#     if report_file:
#         file_extension = os.path.splitext(report_file.filename)[1]
#         unique_filename = f"{patient_data.get('name', 'Unknown')}_{uuid.uuid4()}{file_extension}"
#         file_path = os.path.join(REPORTS_FOLDER, unique_filename)
#         report_file.save(file_path)
#         report_path = file_path
    
#     request_id = str(uuid.uuid4())
    
#     pending_requests.append({
#         'request_id': request_id,
#         'patient': patient_data,
#         'user_location': {'lat': user_lat, 'lng': user_lng},
#         'destination_hospital': destination_hospital,
#         'report_path': report_path,
#         'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     })
    
#     print(f"New request added with ID: {request_id}")
#     return jsonify({"message": "Request received and an ambulance is being dispatched.", "request_id": request_id})

# @app.route('/get_route_details', methods=['POST'])
# def get_route_details():
#     user_lat = float(request.form['latitude'])
#     user_lng = float(request.form['longitude'])
#     patient_data = json.loads(request.form['patient'])
#     selected_hospital_name = request.form.get('hospitalName')
    
#     user_loc = (user_lat, user_lng)
    
#     destination_hospital = None
#     if selected_hospital_name:
#         for hospital in hospitals:
#             if hospital["name"] == selected_hospital_name:
#                 destination_hospital = hospital
#                 break
#     else:
#         shortest_distance = float('inf')
#         for hospital in hospitals:
#             hosp_loc = (hospital["lat"], hospital["lng"])
#             dist = geodesic(user_loc, hosp_loc).km
#             if dist < shortest_distance:
#                 shortest_distance = dist
#                 destination_hospital = hospital
    
#     if not destination_hospital:
#         return jsonify({"error": "No valid hospital found."}), 404
        
#     if destination_hospital['name'] in medchal_hospitals:
#         return jsonify({
#             "error": "Busier",
#             "hospital": destination_hospital,
#             "patient": patient_data,
#             "latitude": user_lat,
#             "longitude": user_lng
#         })

#     coords = [(user_lng, user_lat), (destination_hospital["lng"], destination_hospital["lat"])]
#     try:
#         route = client.directions(coords, profile='driving-car', format='geojson')
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

#     return jsonify({
#         "route": route,
#         "hospital": destination_hospital,
#         "patient": patient_data,
#         "latitude": user_lat,
#         "longitude": user_lng
#     })

# if __name__ == '__main__':
#     app.run(debug=True)




# from flask import Flask, render_template, request, jsonify, redirect, url_for
# import json
# import os
# import openrouteservice
# from openrouteservice import convert
# from geopy.distance import geodesic
# from datetime import datetime
# import time
# import uuid

# app = Flask(__name__)

# ORS_API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6ImVlYzU2MTQ5NDA4ODQ5NjJiYWYwNzE2ZTEwYTEwMjA5IiwiaCI6Im11cm11cjY0In0="
# client = openrouteservice.Client(key=ORS_API_KEY)

# pending_requests = []
# accepted_trips = {}

# REPORTS_FOLDER = 'patient_reports'
# if not os.path.exists(REPORTS_FOLDER):
#     os.makedirs(REPORTS_FOLDER)

# with open("hospitals.json", "r") as f:
#     hospitals = json.load(f)

# medchal_hospitals = [
#     "Leela Multispeciality – Medchal",
#     "VVR Hospital – Medchal",
#     "Dr Raju’s Hospital – Medchal",
#     "Medicity Hospital – Medchal",
#     "Pavithran Hospital – Medchal",
#     "Sankhya Hospitals – Medchal"
# ]

# @app.route('/')
# def home():
#     return render_template("index.html")

# @app.route('/driver')
# def driver_dashboard():
#     return render_template("driver_dashboard.html")

# @app.route('/status/<request_id>')
# def status_page(request_id):
#     return render_template("status.html", request_id=request_id)

# @app.route('/hospitals')
# def get_hospitals():
#     return jsonify(hospitals)

# @app.route('/get_requests')
# def get_requests():
#     is_driver_busy = len(accepted_trips) > 0 and 'arrived_at_hospital' not in [trip['status'] for trip in accepted_trips.values()]
    
#     if is_driver_busy:
#         return jsonify([])
#     else:
#         return jsonify(pending_requests)

# @app.route('/update_trip_status', methods=['POST'])
# def update_trip_status():
#     data = request.get_json()
#     request_id = data.get('request_id')
#     new_status = data.get('status')
    
#     trip = accepted_trips.get(request_id)
#     if trip:
#         trip['status'] = new_status
#         if new_status == 'arrived_at_hospital':
#             trip['dropoff_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#             del accepted_trips[request_id]
#             print(f"Trip {request_id} completed.")
#         return jsonify({"message": f"Trip status updated to {new_status}."})
    
#     return jsonify({"error": "Trip not found."}), 404

# @app.route('/get_trip_status/<request_id>')
# def get_trip_status(request_id):
#     trip = accepted_trips.get(request_id)
#     if trip:
#         return jsonify({
#             'status': trip.get('status', 'enroute_to_patient'),
#             'patient_location': trip.get('user_location'),
#             'destination_hospital': trip.get('destination_hospital'),
#             'patient': trip.get('patient')
#         })
#     return jsonify({"status": "pending", "message": "Awaiting ambulance dispatch."})

# @app.route('/accept_trip', methods=['POST'])
# def accept_trip():
#     data = request.get_json()
#     request_id = data.get('request_id')

#     request_data = None
#     for req in pending_requests:
#         if req['request_id'] == request_id:
#             request_data = req
#             break
    
#     if request_data:
#         pending_requests.remove(request_data)
        
#         user_loc = (request_data['user_location']['lat'], request_data['user_location']['lng'])
#         hosp_loc = (request_data['destination_hospital']['lat'], request_data['destination_hospital']['lng'])
#         coords = [(user_loc[1], user_loc[0]), (hosp_loc[1], hosp_loc[0])]
        
#         try:
#             # Set the profile to a value that provides the most efficient route, which often includes real-time traffic
#             route = client.directions(coords, profile='driving-car', preference='fastest', format='geojson')
#         except Exception as e:
#             print(f"Error getting route for accepted trip: {e}")
#             route = None

#         request_data['status'] = 'enroute_to_patient'
#         request_data['pickup_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#         request_data['route_data'] = route
#         accepted_trips[request_id] = request_data
#         print(f"Driver accepted trip {request_id}")
#         return jsonify({"message": "Trip accepted!", "trip": request_data})
    
#     return jsonify({"error": "Request not found."}), 404

# @app.route('/route', methods=['POST'])
# def get_route():
#     user_lat = float(request.form['latitude'])
#     user_lng = float(request.form['longitude'])
#     patient_data = json.loads(request.form['patient'])
#     selected_hospital_name = request.form.get('hospitalName')
#     report_file = request.files.get('report')

#     user_loc = (user_lat, user_lng)
    
#     destination_hospital = None
#     if selected_hospital_name:
#         for hospital in hospitals:
#             if hospital["name"] == selected_hospital_name:
#                 destination_hospital = hospital
#                 break
#     else:
#         shortest_distance = float('inf')
#         for hospital in hospitals:
#             hosp_loc = (hospital["lat"], hospital["lng"])
#             dist = geodesic(user_loc, hosp_loc).km
#             if dist < shortest_distance:
#                 shortest_distance = dist
#                 destination_hospital = hospital
    
#     if not destination_hospital:
#         return jsonify({"error": "No valid hospital found."}), 404

#     if destination_hospital['name'] in medchal_hospitals:
#         return jsonify({"error": "Busier"})
    
#     report_path = None
#     if report_file:
#         file_extension = os.path.splitext(report_file.filename)[1]
#         unique_filename = f"{patient_data.get('name', 'Unknown')}_{uuid.uuid4()}{file_extension}"
#         file_path = os.path.join(REPORTS_FOLDER, unique_filename)
#         report_file.save(file_path)
#         report_path = file_path
    
#     request_id = str(uuid.uuid4())
    
#     pending_requests.append({
#         'request_id': request_id,
#         'patient': patient_data,
#         'user_location': {'lat': user_lat, 'lng': user_lng},
#         'destination_hospital': destination_hospital,
#         'report_path': report_path,
#         'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     })
    
#     print(f"New request added with ID: {request_id}")
#     return jsonify({"message": "Request received and an ambulance is being dispatched.", "request_id": request_id})

# @app.route('/get_route_details', methods=['POST'])
# def get_route_details():
#     user_lat = float(request.form['latitude'])
#     user_lng = float(request.form['longitude'])
#     patient_data = json.loads(request.form['patient'])
#     selected_hospital_name = request.form.get('hospitalName')
    
#     user_loc = (user_lat, user_lng)
    
#     destination_hospital = None
#     if selected_hospital_name:
#         for hospital in hospitals:
#             if hospital["name"] == selected_hospital_name:
#                 destination_hospital = hospital
#                 break
#     else:
#         shortest_distance = float('inf')
#         for hospital in hospitals:
#             hosp_loc = (hospital["lat"], hospital["lng"])
#             dist = geodesic(user_loc, hosp_loc).km
#             if dist < shortest_distance:
#                 shortest_distance = dist
#                 destination_hospital = hospital
    
#     if not destination_hospital:
#         return jsonify({"error": "No valid hospital found."}), 404
        
#     if destination_hospital['name'] in medchal_hospitals:
#         return jsonify({
#             "error": "Busier",
#             "hospital": destination_hospital,
#             "patient": patient_data,
#             "latitude": user_lat,
#             "longitude": user_lng
#         })

#     coords = [(user_lng, user_lat), (destination_hospital["lng"], destination_hospital["lat"])]
#     try:
#         # Use preference='fastest' to prioritize low-traffic routes
#         route = client.directions(coords, profile='driving-car', preference='fastest', format='geojson')
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

#     return jsonify({
#         "route": route,
#         "hospital": destination_hospital,
#         "patient": patient_data,
#         "latitude": user_lat,
#         "longitude": user_lng
#     })

# if __name__ == '__main__':
#     app.run(debug=True)





from flask import Flask, render_template, request, jsonify, redirect, url_for
import json
import os
import openrouteservice
from openrouteservice import convert
from geopy.distance import geodesic
from datetime import datetime
import time
import uuid

app = Flask(__name__)

# NOTE: You MUST replace this with your own valid OpenRouteService API key.
# A 403 error means this key is invalid or expired.
ORS_API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6ImVlYzU2MTQ5NDA4ODQ5NjJiYWYwNzE2ZTEwYTEwMjA5IiwiaCI6Im11cm11cjY0In0="
client = openrouteservice.Client(key=ORS_API_KEY)

# Global lists to manage requests (in a real app, this would be a database)
pending_requests = []
accepted_trips = {}

REPORTS_FOLDER = 'patient_reports'
if not os.path.exists(REPORTS_FOLDER):
    os.makedirs(REPORTS_FOLDER)

with open("hospitals.json", "r") as f:
    hospitals = json.load(f)

medchal_hospitals = [
    "Leela Multispeciality – Medchal",
    "VVR Hospital – Medchal",
    "Dr Raju’s Hospital – Medchal",
    "Medicity Hospital – Medchal",
    "Pavithran Hospital – Medchal",
    "Sankhya Hospitals – Medchal"
]

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/driver')
def driver_dashboard():
    return render_template("driver_dashboard.html")

@app.route('/status/<request_id>')
def status_page(request_id):
    return render_template("status.html", request_id=request_id)

@app.route('/hospitals')
def get_hospitals():
    """Returns the list of hospitals for the dropdown."""
    return jsonify(hospitals)

@app.route('/get_requests')
def get_requests():
    """Endpoint for drivers to get pending requests."""
    is_driver_busy = len(accepted_trips) > 0 and 'arrived_at_hospital' not in [trip['status'] for trip in accepted_trips.values()]
    
    if is_driver_busy:
        return jsonify([])
    else:
        return jsonify(pending_requests)

@app.route('/update_trip_status', methods=['POST'])
def update_trip_status():
    data = request.get_json()
    request_id = data.get('request_id')
    new_status = data.get('status')
    
    trip = accepted_trips.get(request_id)
    if trip:
        trip['status'] = new_status
        if new_status == 'arrived_at_hospital':
            trip['dropoff_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            del accepted_trips[request_id]
            print(f"Trip {request_id} completed.")
        return jsonify({"message": f"Trip status updated to {new_status}."})
    
    return jsonify({"error": "Trip not found."}), 404

@app.route('/get_trip_status/<request_id>')
def get_trip_status(request_id):
    trip = accepted_trips.get(request_id)
    if trip:
        return jsonify({
            'status': trip.get('status', 'enroute_to_patient'),
            'patient_location': trip.get('user_location'),
            'destination_hospital': trip.get('destination_hospital'),
            'patient': trip.get('patient')
        })
    return jsonify({"status": "pending", "message": "Awaiting ambulance dispatch."})

@app.route('/accept_trip', methods=['POST'])
def accept_trip():
    data = request.get_json()
    request_id = data.get('request_id')

    request_data = None
    for req in pending_requests:
        if req['request_id'] == request_id:
            request_data = req
            break
    
    if request_data:
        pending_requests.remove(request_data)
        
        user_loc = (request_data['user_location']['lat'], request_data['user_location']['lng'])
        hosp_loc = (request_data['destination_hospital']['lat'], request_data['destination_hospital']['lng'])
        coords = [(user_loc[1], user_loc[0]), (hosp_loc[1], hosp_loc[0])]
        
        try:
            route = client.directions(coords, profile='driving-car', preference='fastest', format='geojson')
        except Exception as e:
            print(f"Error getting route for accepted trip: {e}")
            route = None

        request_data['status'] = 'enroute_to_patient'
        request_data['pickup_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        request_data['route_data'] = route
        accepted_trips[request_id] = request_data
        print(f"Driver accepted trip {request_id}")
        return jsonify({"message": "Trip accepted!", "trip": request_data})
    
    return jsonify({"error": "Request not found."}), 404

@app.route('/route', methods=['POST'])
def get_route():
    user_lat = float(request.form['latitude'])
    user_lng = float(request.form['longitude'])
    patient_data = json.loads(request.form['patient'])
    selected_hospital_name = request.form.get('hospitalName')
    report_file = request.files.get('report')

    user_loc = (user_lat, user_lng)
    
    destination_hospital = None
    if selected_hospital_name:
        for hospital in hospitals:
            if hospital["name"] == selected_hospital_name:
                destination_hospital = hospital
                break
    else:
        shortest_distance = float('inf')
        for hospital in hospitals:
            hosp_loc = (hospital["lat"], hospital["lng"])
            dist = geodesic(user_loc, hosp_loc).km
            if dist < shortest_distance:
                shortest_distance = dist
                destination_hospital = hospital
    
    if not destination_hospital:
        return jsonify({"error": "No valid hospital found."}), 404

    if destination_hospital['name'] in medchal_hospitals:
        return jsonify({"error": "Busier"})
    
    report_path = None
    if report_file:
        file_extension = os.path.splitext(report_file.filename)[1]
        unique_filename = f"{patient_data.get('name', 'Unknown')}_{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(REPORTS_FOLDER, unique_filename)
        report_file.save(file_path)
        report_path = file_path
    
    request_id = str(uuid.uuid4())
    
    pending_requests.append({
        'request_id': request_id,
        'patient': patient_data,
        'user_location': {'lat': user_lat, 'lng': user_lng},
        'destination_hospital': destination_hospital,
        'report_path': report_path,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })
    
    print(f"New request added with ID: {request_id}")
    return jsonify({"message": "Request received and an ambulance is being dispatched.", "request_id": request_id})

@app.route('/get_route_details', methods=['POST'])
def get_route_details():
    user_lat = float(request.form['latitude'])
    user_lng = float(request.form['longitude'])
    patient_data = json.loads(request.form['patient'])
    selected_hospital_name = request.form.get('hospitalName')
    
    user_loc = (user_lat, user_lng)
    
    destination_hospital = None
    if selected_hospital_name:
        for hospital in hospitals:
            if hospital["name"] == selected_hospital_name:
                destination_hospital = hospital
                break
    else:
        shortest_distance = float('inf')
        for hospital in hospitals:
            hosp_loc = (hospital["lat"], hospital["lng"])
            dist = geodesic(user_loc, hosp_loc).km
            if dist < shortest_distance:
                shortest_distance = dist
                destination_hospital = hospital
    
    if not destination_hospital:
        return jsonify({"error": "No valid hospital found."}), 404
        
    if destination_hospital['name'] in medchal_hospitals:
        return jsonify({
            "error": "Busier",
            "hospital": destination_hospital,
            "patient": patient_data,
            "latitude": user_lat,
            "longitude": user_lng
        })

    coords = [(user_lng, user_lat), (destination_hospital["lng"], destination_hospital["lat"])]
    try:
        route = client.directions(coords, profile='driving-car', preference='fastest', format='geojson')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "route": route,
        "hospital": destination_hospital,
        "patient": patient_data,
        "latitude": user_lat,
        "longitude": user_lng
    })

if __name__ == '__main__':
    app.run(debug=True)