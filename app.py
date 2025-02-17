import cv2
import os
from flask import Flask, request, render_template, send_from_directory,jsonify
from datetime import date, datetime
import pandas as pd
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
import threading
import time


# Defining Flask App
app = Flask(__name__)

nimgs = 50
max_unknown_images = 10
max_tailgating_images = 10

# Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

# Load the pre-trained FaceNet model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Initialize MTCNN for face detection
mtcnn = MTCNN(keep_all=True, device=device)

# Ensure necessary directories exist
os.makedirs('Attendance', exist_ok=True)
os.makedirs('static', exist_ok=True)
os.makedirs('static/faces', exist_ok=True)
os.makedirs('Attendance/Alerts', exist_ok=True)

# Initialize attendance file if it doesn't exist
if not os.path.exists('Attendance/Recognized-list.csv'):
    df = pd.DataFrame(columns=['Roll no', 'Name'])
    df.to_csv('Attendance/Recognized-list.csv', index=False)

def update_attendance(file_path):
    df = pd.read_csv(file_path)
    today = datetime.today().strftime('%Y-%m-%d')
    if today not in df.columns:
        df[today] = '-'
    df.to_csv(file_path, index=False)

file_path = 'Attendance/Recognized-list.csv'
update_attendance(file_path)

def reset_recognized_list(file_path):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        attendance_columns = [col for col in df.columns if col not in ['Roll no', 'Name']]
        if len(attendance_columns) > 7:
            df = df[['Roll no', 'Name']]
            df.to_csv(file_path, index=False)

reset_recognized_list(file_path)

def totalreg():
    return len(os.listdir('static/faces'))

def extract_face_embeddings(img):
    with torch.no_grad():
        faces, _ = mtcnn(img, return_prob=True)
        if faces is not None:
            faces = faces.to(device)
            embeddings = model(faces)
            return embeddings
    return None

def cache_embeddings(directory):
    embeddings = []
    for person in os.listdir(directory):
        person_dir = os.path.join(directory, person)
        if os.path.isdir(person_dir):
            for file in os.listdir(person_dir):
                if file.endswith(('jpg', 'jpeg', 'png')):
                    image_path = os.path.join(person_dir, file)
                    img = Image.open(image_path)
                    embedding = extract_face_embeddings(img)
                    if embedding is not None:
                        embeddings.append((embedding[0].cpu(), person))
                        del embedding  # Free up GPU memory
                        torch.cuda.empty_cache()  # Empty the cache if using GPU
    return embeddings

known_faces_dir = 'static/faces'
known_embeddings = cache_embeddings(known_faces_dir)

def compare_faces(embedding, known_embeddings, threshold=0.80):
    distances = [(torch.dist(embedding, known_embedding).item(), label) for known_embedding, label in known_embeddings]
    min_distance, label = min(distances, key=lambda x: x[0])
    return min_distance < threshold, label if min_distance < threshold else None

# Initialize face tracker
tracker = cv2.legacy.TrackerKCF_create() if hasattr(cv2, 'legacy') else cv2.TrackerKCF_create()

init_tracker = False
track_box = None
track_label = None

unknown_counter = 0
known_counter = 0
tailgating_counter = 0

def reset_counters():
    global unknown_counter, known_counter
    unknown_counter = 0
    known_counter = 0


def daily_reset():
    while True:
        now = datetime.now()
        # Calculate seconds until next midnight
        seconds_until_midnight = ((24 - now.hour - 1) * 3600) + ((60 - now.minute - 1) * 60) + (60 - now.second)
        time.sleep(seconds_until_midnight)
        reset_counters()


# Ensure the tailgating directory exists
tailgating_counter = 0
max_tailgating_images = 10  # Set the maximum number of images to save

def save_tailgating_image(frame):
    global tailgating_counter

    today = datetime.today().strftime('%Y-%m-%d')
    current_time = datetime.now().strftime('%H-%M-%S')
    
    tailgating_dir = os.path.join('Attendance/Tailgate', today)
    os.makedirs(tailgating_dir, exist_ok=True)
    
    file_path = os.path.join(tailgating_dir, f'tailgating_{today}_{current_time}.jpg')
    
    print(f"Saving tailgating image to {file_path}")
    if frame is None or frame.size == 0:
        print("Error: Frame is empty or None")
        return

    try:
        cv2.imwrite(file_path, frame)
        tailgating_counter += 1
        print(f"Image successfully saved to {file_path}")
    except Exception as e:
        print(f"Failed to save image: {e}")

def detect_tailgating(faces, frame):
    if len(faces) < 2:
        return False

    if len(faces) >= 2:
        print("Tailgating detected!")
        cv2.putText(frame, 'TAILGATING', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4, cv2.LINE_AA)
        save_tailgating_image(frame)
        return True
    return False

# Initialize MTCNN for face detection
mtcnn = MTCNN(keep_all=True, device=device)

def process_frame(frame):
    success = False
    global init_tracker, track_box, track_label, unknown_counter
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    embeddings = extract_face_embeddings(img)
    faces, _ = mtcnn.detect(frame)

    if faces is not None and embeddings is not None:
        print(f"Detected faces: {faces}")
        detect_tailgating(faces, frame)  # Check for tailgating
        detected_labels = []

        for embedding in embeddings:
            similar, label = compare_faces(embedding.cpu(), known_embeddings)
            if similar:
                detected_labels.append(label)
            else:
                print("Detected an unknown face")
                if not init_tracker:
                    x1, y1, x2, y2 = tuple(map(int, faces[0]))
                    track_box = (x1, y1, x2 - x1, y2 - y1)
                    tracker.init(frame, track_box)
                    track_label = "UNIDENTIFIED PERSON"
                    init_tracker = True
                return "Undetected", None  # Correct return value for undetected faces

        if detected_labels:
            print(f"Detected labels: {detected_labels}")
            if not init_tracker:
                x1, y1, x2, y2 = tuple(map(int, faces[0]))
                track_box = (x1, y1, x2 - x1, y2 - y1)
                tracker.init(frame, track_box)
                track_label = detected_labels[0]
                init_tracker = True
            return True, detected_labels
    else:
        return False, None  # No faces detected


def extract_attendance(filename):
    df = pd.read_csv(filename)
    today = datetime.today().strftime('%Y-%m-%d')
    
    # Ensure today's column exists
    if today not in df.columns:
        df[today] = '-'
        df.to_csv(filename, index=False)  # Save the updated CSV

    names = df['Name']
    rolls = df['Roll no']
    times = df[today]
    l = len(df)
    return names, rolls, times, l

def mark_attendance(file_path, name, is_present=True):
    df = pd.read_csv(file_path)
    today = datetime.today().strftime('%Y-%m-%d')
    
    # Ensure today's column exists
    if today not in df.columns:
        df[today] = '-'
    
    current_time = datetime.now().strftime('%H:%M:%S')
    df.loc[df['Name'] == name, today] = current_time if is_present else '-'
    df.to_csv(file_path, index=False)


def add_new_name(file_path, roll_no, name):
    df = pd.read_csv(file_path)
    if roll_no not in df['Roll no'].values:
        new_row = {'Roll no': roll_no, 'Name': name}
        for column in df.columns:
            if column not in ['Roll no', 'Name']:
                new_row[column] = '-'
        new_df = pd.DataFrame(new_row, index=[0])
        df = pd.concat([df, new_df], ignore_index=True)
        df.to_csv(file_path, index=False)

def check_attendance(file_path, name):
    df = pd.read_csv(file_path)
    today = datetime.today().strftime('%Y-%m-%d')
    if name in df['Name'].values:
        attendance_status = df.loc[df['Name'] == name, today].values[0]
        return attendance_status
    else:
        return f"No record found for Name {name}"

def unknown_update(frame):
    global unknown_counter

    print("inside unknown update function")
    today = datetime.today().strftime('%Y-%m-%d')
    current_time = datetime.now().strftime('%H-%M-%S')
    
    Alerts_dir = os.path.join('Attendance/Alerts',today)
    os.makedirs(Alerts_dir, exist_ok=True)
    
    file_path = os.path.join(Alerts_dir, f'unknown_{today}_{current_time}.jpg')
    
    print(f"Saving unidentified face to {file_path}")
    if frame is None or frame.size == 0:
        print("Error: Frame is empty or None")
        return

    try:
        cv2.imwrite(file_path, frame)
        unknown_counter += 1
        print(f"Image successfully saved to {file_path}")
    except Exception as e:
        print(f"Failed to save image: {e}")

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/index')
def index():
    names, rolls, times, l = extract_attendance(file_path)
    return render_template('index.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/api/total_registered_employees', methods=['GET'])
def total_registered_employees():
    total_count = totalreg()
    return jsonify({'total_registered_employees': total_count})

@app.route('/start', methods=['GET'])
def start():
    global init_tracker, track_box, track_label, unknown_counter, tailgating_counter, known_counter
    names, rolls, times, l = extract_attendance(file_path)
    alert = 0
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
                
        detected, detected_labels = process_frame(frame)
        print(f"Detection result: {detected}")
        if detected == True:
            for label in detected_labels:
                print(f"Detected: {label}")
        elif detected == False:
            print("No face detected")
        elif detected == "Undetected":
            print("Unidentified person")
            
        print(f"init_tracker state: {init_tracker}")
        if init_tracker:
            success, box = tracker.update(frame)
            print(f"Tracker update success: {success}, Box: {box}")
            if success:
                x1, y1, w, h = tuple(map(int, box))
                x2, y2 = x1 + w, y1 + h
                if detected == "Undetected":
                    alert += 1
                    print("putting bounding box")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, track_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    if y2 > y1 and x2 > x1:  # Ensure valid coordinates
                        unknown_update(frame[y1:y2, x1:x2])
                    init_tracker = False
                    print("Tracker reset for unknown person")
                    
                elif detected == True:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, track_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    name_part = track_label.split('_')[0]            
                    if check_attendance(file_path, name_part) == '-':
                        mark_attendance(file_path, name_part, True) 
                        known_counter+=1                                   

            else:
                init_tracker = False

        cv2.imshow('Security', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    names, rolls, times, l = extract_attendance(file_path)
    return render_template('index.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, unknown_counter=unknown_counter,known_counter=known_counter,tailgating_counter=tailgating_counter)

@app.route('/register')
def register():
    return render_template('index2.html')

@app.route('/add', methods=['POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/'+newusername+'_'+str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    i, j = 0, 0
    cap = cv2.VideoCapture(0)
    while 1:
        _, frame = cap.read()
        faces, _ = mtcnn.detect(frame)
        if faces is not None:
            for (x1, y1, x2, y2) in faces:
                x = int(x1)
                y = int(y1)
                w = int(x2 - x1)
                h = int(y2 - y1)

                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
                cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
                if j % 5 == 0:
                    name = newusername+'_'+str(i)+'.jpg'
                    if i == 1:
                        userface = name
                    cv2.imwrite(userimagefolder+'/'+name, frame[y:y+h, x:x+w])
                    i += 1
                j += 1
            if j == nimgs*5:
                break
            cv2.imshow('Adding new User', frame)
            if cv2.waitKey(1) == 27:
                break
    cap.release()
    cv2.destroyAllWindows()
    add_new_name(file_path, newuserid, newusername)
    names, rolls, times, l = extract_attendance(file_path)
    return render_template('index2.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/for_password')
def for_password():
    return render_template('forgot_pass.html')

@app.route('/otp')
def otp():
    return render_template('forgot_pass_otp.html')

@app.route('/paass_change')
def paass_change():
    return render_template('change_pass.html')

def datetoday2():
    return datetime.now().strftime("%d %B, %Y")

def get_todays_total_count(file_path):
    df = pd.read_csv(file_path)
    today = datetime.today().strftime('%Y-%m-%d')
    print(f"Today's date: {today}")
    if today in df.columns:
        count = df[today].notna().sum()
        print(f"Count for {today}: {count}")
        return count
    print(f"No data for {today}")
    return 0

@app.route('/dashboard', methods=['GET'])
def dashboard():
    global unknown_counter, known_counter, tailgating_counter
    names, rolls, times, l = extract_attendance(file_path)
    current_date = datetoday2()
    total_registered_employees = totalreg()  # Get total number of registered employees
    todays_total_count = get_todays_total_count(file_path)
    unknown = unknown_counter
    tailgating = tailgating_counter
    return render_template('dashboard.html', names=names, rolls=rolls, times=times, l=l, totalreg=total_registered_employees, datetoday2=current_date, unknown_counter=unknown, known_counter=todays_total_count, tailgating_counter=tailgating)

ATTENDANCE_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Attendance', 'Alerts')

@app.route('/alert')
def alert():
    date_str = request.args.get('date')
    if date_str:
        date_folder = os.path.join(ATTENDANCE_FOLDER, date_str)
        if os.path.exists(date_folder):
            images = os.listdir(date_folder)
            images = [f"{date_str}/{img}" for img in images if img.lower().endswith(('png', 'jpg', 'jpeg', 'gif'))]
        else:
            images = []
    else:
        date_str = datetime.now().strftime('%Y-%m-%d')
        images = []

    return render_template('alert.html', images=images, selected_date=date_str)

@app.route('/attendance/alerts/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(ATTENDANCE_FOLDER, filename)

TAILGATING_IMG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Attendance', 'Tailgate')

@app.route('/tailgate')
def tailgate():
    date_str = request.args.get('date')
    if date_str:
        date_folder = os.path.join(TAILGATING_IMG, date_str)
        if os.path.exists(date_folder):
            images = os.listdir(date_folder)
            images = [f"{date_str}/{img}" for img in images if img.lower().endswith(('png', 'jpg', 'jpeg', 'gif'))]
        else:
            images = []
    else:
        date_str = datetime.now().strftime('%Y-%m-%d')
        images = []

    return render_template('tailgating.html', images=images, selected_date=date_str)

@app.route('/attendance/tailgate/<path:filename>')
def uploaded_filet(filename):
    return send_from_directory(TAILGATING_IMG, filename)

entries = [
    {'id': 1, 'employee_id': '9979', 'employee_name': 'Ritika PalChaudhuri', 'time': '08:15'},
    # Add more entries here
]



@app.route('/get_data')
def get_data():
    data = {
        'date': datetime.datetime.now().strftime("%d %B, %Y"),
        'total_registered_employees': 10,
        'todays_total_count': 9,
        'registered_entries': 7,
        'unidentified_entries': 2,
        'tailgating_frames': 2,
        'entries': entries
    }
    return jsonify(data)



if __name__ == '__main__':
    threading.Thread(target=daily_reset, daemon=True).start()
    app.run(debug=True)
