import os
import json 
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import openai
import httpx # <-- NEW IMPORT for asynchronous HTTP requests

# Load environment variables (like OPENAI_API_KEY) from .env file
load_dotenv()

app = FastAPI()
# Initialize the OpenAI client

OPENAI_KEY  = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    print("FATAL ERROR: OPENAI_API_KEY not found in environment.")
openai.api_key = OPENAI_KEY
# --- FHIR Configuration ---
# CRITICAL: Using the public HAPI FHIR Test Server for data retrieval
FHIR_BASE_URL = "https://hapi.fhir.org/baseR4" 
# --------------------------


# ----------------- Appointment Store & Clinic Data -----------------
appointments = []

DOCTOR_AVAILABILITY = [
    {"doctor": "Dr. Alex Johnson", "specialty": "General Practice", "times": ["09:00", "10:00", "11:00", "14:00", "15:00"]},
    {"doctor": "Dr. Sarah Chen", "specialty": "Pediatrics", "times": ["10:30", "11:30", "13:30", "14:30", "15:30"]},
]

# --- Helper Functions ---

def is_available(time_slot: str) -> bool:
    """
    Checks if a time slot is available AND ensures the date is not in the past.
    Returns True only if available and in the future.
    """
    try:
        appointment_time = datetime.strptime(time_slot, "%Y-%m-%d %H:%M")
        if appointment_time <= datetime.now():
            return False 
    except ValueError:
        return False 

    return not any(appt["timeSlot"] == time_slot for appt in appointments)

def schedule_appointment(patient_name: str, time_slot: str) -> dict:
    """Schedules a new appointment."""
    appt = {"patientName": patient_name, "timeSlot": time_slot}
    appointments.append(appt)
    return appt

def cancel_appointment(patient_name: str, time_slot: str) -> bool:
    """Cancels an existing appointment."""
    for appt in appointments:
        if appt["patientName"] == patient_name and appt["timeSlot"] == time_slot:
            appointments.remove(appt)
            return True
    return False

# --- New FHIR Data Retrieval Function ---
# In gpt_back.py (near line 60):

async def get_patient_records(patient_id: str) -> dict:
    """
    Retrieves patient records using a FHIR SEARCH by identifier, 
    which is more robust than a direct read.
    """
    if not patient_id:
        return {"status": "error", "message": "Patient ID is required."}

    # Use a client context manager for safe async requests
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            # 1. FIX: Perform a SEARCH by identifier instead of a direct read by ID.
            # This handles both logical IDs and external MRNs when the server is configured
            patient_url = f"{FHIR_BASE_URL}/Patient?_id={patient_id}" # Search by logical ID first
            patient_response = await client.get(patient_url)
            patient_response.raise_for_status()
            
            patient_bundle = patient_response.json()

            # 2. Extract Data from the Search Bundle
            
            # Check if the bundle returned any results
            if not patient_bundle.get("entry"):
                # If the first search fails, try searching by external identifier (MRN)
                patient_url_id = f"{FHIR_BASE_URL}/Patient?identifier={patient_id}"
                patient_response_id = await client.get(patient_url_id)
                patient_response_id.raise_for_status()
                patient_bundle = patient_response_id.json()
                
                if not patient_bundle.get("entry"):
                    return {"status": "error", "message": f"Record not found for ID {patient_id}. Please ensure the ID is correct."}


            # Get the first Patient resource from the bundle
            patient_resource = patient_bundle["entry"][0]["resource"]

            # 3. Process Demographics (Unchanged Logic)
            first_name = patient_resource.get('name', [{}])[0].get('given', [''])[0]
            last_name = patient_resource.get('name', [{}])[0].get('family', 'Unknown')
            
            patient_name = f"{first_name} {last_name}".strip()
            if not patient_name:
                patient_name = f"Patient {patient_id}"

            demographics = (
                f"Gender: {patient_resource.get('gender')}, DOB: {patient_resource.get('birthDate', 'N/A')}"
            )
            
            # 4. Get Recent Conditions (Unchanged Logic - requires the resource's logical ID)
            # Use the logical ID from the resource we just found
            logical_id = patient_resource.get("id")
            conditions_url = f"{FHIR_BASE_URL}/Condition?patient={logical_id}&_sort=-recorded-date&_count=3"
            conditions_response = await client.get(conditions_url)
            conditions_bundle = conditions_response.json()
            
            conditions_list = []
            if conditions_bundle.get("entry"):
                for entry in conditions_bundle["entry"]:
                    condition = entry["resource"]
                    code = condition.get('code', {}).get('coding', [{}])[0].get('display', 'N/A')
                    conditions_list.append(code)

            # 5. Overlay Local Appointment Data (Unchanged Logic)
            name_for_local_lookup = patient_name 
            
            local_appts = [
                appt["timeSlot"] for appt in appointments 
                if appt["patientName"].lower() == name_for_local_lookup.lower()
            ]

            return {
                "status": "success",
                "patient_name": patient_name,
                "demographics": demographics,
                "recent_conditions": conditions_list,
                "local_appointments": local_appts
            }

        except httpx.HTTPStatusError as e:
            # This catch will now only fire for true server errors (5xx) or non-404 client errors
            return {"status": "error", "message": f"FHIR Server Error ({e.response.status_code}): Server/Network issue."}
        except Exception as e:
            return {"status": "error", "message": f"An unexpected error occurred: {e}"}
            
# ----------------- Tools for AI (OpenAI Function Definitions) -----------------
tools = [
    # ... Existing schedule_appointment tool ...
    {
        "type": "function",
        "function": {
            "name": "schedule_appointment",
            "description": "Book a new appointment for a patient. Requires patient name and the full time slot (YYYY-MM-DD HH:mm).",
            "parameters": {
                "type": "object",
                "properties": {
                    "patientName": {"type": "string"},
                    "timeSlot": {"type": "string", "description": "Format YYYY-MM-DD HH:mm"},
                },
                "required": ["patientName", "timeSlot"]
            },
        },
    },
    # ... Existing cancel_appointment tool ...
    {
        "type": "function",
        "function": {
            "name": "cancel_appointment",
            "description": "Cancel an existing appointment. ONLY use this function when the user explicitly asks to remove, cancel, or delete a prior booking. Requires patient name and the full time slot (YYYY-MM-DD HH:mm) of the appointment to be removed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patientName": {"type": "string"},
                    "timeSlot": {"type": "string"},
                },
                "required": ["patientName", "timeSlot"]
            },
        },
    },
    # --- New FHIR Retrieval Tool ---
    {
        "type": "function",
        "function": {
            "name": "get_patient_records",
            "description": "Retrieves the patient's medical records (demographics and recent conditions) using their external Patient ID or MRN. Use this when the user asks to look up a patient's history or records.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patientId": {"type": "string", "description": "The unique Patient ID (e.g., MRN or FHIR ID)."},
                },
                "required": ["patientId"]
            },
        },
    },
]

# ----------------- Chatbot Route -----------------
@app.post("/api/chatbot")
async def chatbot(request: Request):
    body = await request.json()
    user_message = body.get("message", "")
    history = body.get("history", []) 

    system_prompt = (
        "You are a helpful clinic assistant. Your primary goal is to manage appointments and retrieve patient records. "
        "Analyze the full conversation history. You must respond in plain text, do not use markdown formatting in your replies."
    )
    
    messages = [{"role": "system", "content": system_prompt}] 
    messages.extend(history) 
    messages.append({"role": "user", "content": user_message}) 
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages, 
            tools=tools
        )
    except openai.APIConnectionError as e:
        print(f"OpenAI API Connection Error: {e}")
        return JSONResponse({"reply": "⚠️ Critical Error: Could not connect to the OpenAI API. Check network/API key."}, status_code=500)
    except Exception as e:
        print(f"Unexpected OpenAI Error: {e}")
        return JSONResponse({"reply": "⚠️ An unexpected error occurred on the server."}, status_code=500)


    msg = response.choices[0].message

    # If the model wants to call a tool
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        tool_call = msg.tool_calls[0]
        fn_name = tool_call.function.name
        
        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            reply = "⚠️ Internal Error: Failed to parse function arguments from AI."
            return JSONResponse({"reply": reply})

        # --- SCHEDULE APPOINTMENT ---
        if fn_name == "schedule_appointment":
            patient = args.get("patientName")
            time = args.get("timeSlot")
            
            if not patient or not time:
                reply = "I lack the patient name or time slot to book the appointment. Please confirm both."
                return JSONResponse({"reply": reply})

            if is_available(time):
                appt = schedule_appointment(patient, time)
                reply = f"✅ Appointment booked for {appt['patientName']} at {appt['timeSlot']}."
            else:
                try:
                    appt_time = datetime.strptime(time, "%Y-%m-%d %H:%M")
                    if appt_time <= datetime.now():
                        reply = "Sorry, you cannot book appointments in the past."
                    else:
                        reply = "Sorry, that slot is already taken."
                except ValueError:
                    reply = "The date/time format is invalid. Please use YYYY-MM-DD HH:mm."

        # --- CANCEL APPOINTMENT ---
        elif fn_name == "cancel_appointment":
            patient = args.get("patientName")
            time = args.get("timeSlot")
            
            if not patient or not time:
                reply = "I lack the patient name or time slot to cancel the appointment. Please confirm both."
                return JSONResponse({"reply": reply})

            if cancel_appointment(patient, time):
                reply = f"🗑️ Appointment canceled for {patient} at {time}."
            else:
                reply = f"⚠️ No matching appointment found for {patient} at {time}."
                
        # --- RETRIEVE FHIR RECORDS ---
        elif fn_name == "get_patient_records":
            patient_id = args.get("patientId")
            
            if not patient_id:
                reply = "I need the patient's ID or Medical Record Number to retrieve the records."
                return JSONResponse({"reply": reply})

            record_data = await get_patient_records(patient_id)
            
            if record_data["status"] == "success":
                conditions_str = ", ".join(record_data["recent_conditions"]) if record_data["recent_conditions"] else "None recorded."
                appts_str = ", ".join(record_data["local_appointments"])
                if not appts_str:
                    appts_str = "No appointments currently booked."

                reply = (
                    f"Patient Record Found for {record_data['patient_name']} (ID: {patient_id}). "
                    f"Demographics: {record_data['demographics']}. "
                    f"Recent Conditions: {conditions_str}. "
                    f"Current Appointments: {appts_str}."
                )
            else:
                reply = f"Error retrieving record: {record_data['message']}"

        else:
            reply = "⚠️ Unknown tool requested."
    else:
        # Normal conversation (Uses dot notation)
        reply = msg.content if msg.content else "Sorry, I didn’t understand."

    return JSONResponse({"reply": reply})

# ----------------- Standard Appointment Routes -----------------
@app.get("/api/appointments")
async def get_appointments(patient_name: str = None):
    """
    Returns live appointments ONLY when filtered by the patient's name.
    """
    if patient_name:
        filtered_appointments = [
            appt for appt in appointments 
            if appt["patientName"].lower() == patient_name.lower()
        ]
        return JSONResponse({"appointments": filtered_appointments})
    else:
        return JSONResponse({"appointments": [], "message": "Please provide your name to view your appointments."})

@app.get("/api/availability")
async def get_availability():
    """Returns general clinic availability and doctor names."""
    return JSONResponse({"availability": DOCTOR_AVAILABILITY})

@app.post("/api/appointments")
async def add_appointment(request: Request):
    """Allows direct creation of an appointment via POST request (outside of chatbot)."""
    data = await request.json()
    patientName = data.get("patientName")
    timeSlot = data.get("timeSlot")

    if not patientName or not timeSlot:
        return JSONResponse({"status": "error", "message": "Missing details"}, status_code=400)

    if is_available(timeSlot):
        appt = schedule_appointment(patientName, timeSlot)
        return JSONResponse({"status": "success", "message": "Appointment booked.", "appointment": appt})
    else:
        try:
            appt_time = datetime.strptime(timeSlot, "%Y-%m-%d %H:%M")
            if appt_time <= datetime.now():
                error_message = "Appointment date is in the past."
            else:
                error_message = "Time slot already booked."
        except ValueError:
            error_message = "Invalid date/time format."

        return JSONResponse({"status": "error", "message": error_message}, status_code=409)

# ----------------- CORS Middleware Setup -----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)