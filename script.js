let map = L.map('map').setView([17.385044, 78.486671], 12);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom: 19
}).addTo(map);

let routeLayer = [];

const hospitals = [
  "Osmania General Hospital Hyderabad",
  "Apollo Hospital Jubilee Hills Hyderabad",
  "KIMS Hospital Begumpet Hyderabad",
  "Care Hospital Banjara Hills Hyderabad",
  "Sunshine Hospital Secunderabad",
  "Aarogya Hospital Medchal",
  "Sri Harsha Hospital Medchal",
  "Sree Lakshmi Hospital Medchal",
  "Ramesh Hospital Shadnagar",
  "Avni Hospital Rangareddy",
  "Sneha Hospital Rangareddy"
];

function showPatientForm(callback) {
  const formHtml = `
    <div id="patientForm" style="
      position: fixed; top: 30px; left: 50%; transform: translateX(-50%);
      background: #fff; padding: 15px; border-radius: 10px;
      box-shadow: 0 0 10px rgba(0,0,0,0.3); z-index: 9999;
    ">
      <h3 style="margin-top:0;">Patient Info</h3>
      <input placeholder="Name" id="pname" style="margin:5px;"><br>
      <input placeholder="Age" id="page" style="margin:5px;"><br>
      <input placeholder="Issue" id="pissue" style="margin:5px;"><br>
      <input placeholder="Blood Group" id="pblood" style="margin:5px;"><br>
      <button onclick="submitPatientForm()">Submit</button>
    </div>
  `;
  document.body.insertAdjacentHTML('beforeend', formHtml);

  window.submitPatientForm = function () {
    const name = document.getElementById("pname").value;
    const age = document.getElementById("page").value;
    const issue = document.getElementById("pissue").value;
    const blood = document.getElementById("pblood").value;

    if (!name || !age || !issue || !blood) {
      alert("Please fill all fields.");
      return;
    }

    document.getElementById("patientForm").remove();
    callback({ name, age, issue, blood_group: blood });
  };
}

navigator.geolocation.getCurrentPosition((position) => {
  const userLat = position.coords.latitude;
  const userLng = position.coords.longitude;

  const source = {
    lat: userLat,
    lng: userLng
  };

  L.marker([source.lat, source.lng]).addTo(map).bindPopup("🚑 Your Location").openPopup();

  showPatientForm((patientInfo) => {
    let bestTime = Infinity;
    let bestHospital = null;
    let bestRouteData = null;
    let processed = 0;

    hospitals.forEach(hospital => {
      fetch(`https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(hospital)}`)
        .then(response => response.json())
        .then(results => {
          if (results.length > 0) {
            const destination = [parseFloat(results[0].lat), parseFloat(results[0].lon)];

            fetch('http://localhost:5000/route', {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                source: source,
                destination: destination
              })
            })
              .then(res => res.json())
              .then(data => {
                processed++;

                if (!data.error && data.duration < bestTime) {
                  bestTime = data.duration;
                  bestHospital = destination;
                  bestRouteData = {
                    geometry: data.geometry,
                    distance: data.distance,
                    duration: data.duration
                  };
                }

                if (processed === hospitals.length && bestRouteData) {
                  // Clear previous routes
                  routeLayer.forEach(layer => map.removeLayer(layer));
                  routeLayer = [];

                  let bestRoute = L.geoJSON(bestRouteData.geometry, {
                    style: { color: 'green', weight: 6 }
                  }).addTo(map);
                  routeLayer.push(bestRoute);

                  let coords = bestRouteData.geometry.coordinates;
                  let midIndex = Math.floor(coords.length / 2);
                  let midpoint = [coords[midIndex][1], coords[midIndex][0]];

                  L.popup()
                    .setLatLng(midpoint)
                    .setContent("🚑 <b>This is the smartest route to reach the hospital quickly with least traffic.</b>")
                    .openOn(map);

                  L.marker(bestHospital).addTo(map).bindPopup("🏥 Nearest Optimal Hospital").openPopup();

                  alert("✅ Ambulance dispatched to nearest hospital with least traffic!");

                  // Log trip to backend
                  fetch("http://localhost:5000/log_trip", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                      name: patientInfo.name,
                      age: patientInfo.age,
                      issue: patientInfo.issue,
                      blood_group: patientInfo.blood_group,
                      source_lat: source.lat,
                      source_lng: source.lng,
                      dest_lat: bestHospital[0],
                      dest_lng: bestHospital[1],
                      distance: bestRouteData.distance,
                      duration: bestRouteData.duration
                    })
                  }).then(res => res.json())
                    .then(res => console.log("Trip logged:", res));
                }
              });
          } else {
            processed++;
          }
        });
    });
  });
});
