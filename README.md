This project is an advanced security and attendance management system leveraging deep learning and computer vision technologies. The system is designed to monitor and record the attendance of employees using facial recognition. Built using Flask for the backend, the system integrates the FaceNet model for facial recognition, allowing it to identify individuals with high accuracy. It also employs the MTCNN (Multi-task Cascaded Convolutional Networks) for efficient face detection in real-time video streams.

**TOOLS USED:**
Flask: A lightweight web framework used to create the web application, providing routing, templating, and API capabilities.
OpenCV: A powerful computer vision library used for capturing and processing video frames, detecting faces, and handling image-related tasks such as saving and displaying images.
PyTorch: A deep learning framework used to load and run the FaceNet model for facial recognition. It enables efficient processing of face embeddings and comparisons.
FaceNet: A pre-trained deep learning model that generates high-dimensional embeddings of faces, which are used for identifying and verifying individuals.
MTCNN (Multi-task Cascaded Convolutional Networks): A face detection algorithm used to accurately locate faces in video frames before they are processed by the FaceNet model.
Pandas: A data manipulation library used to handle attendance records, including reading, updating, and managing CSV files.

**FEATURES:**
Real-time Face Detection and Recognition: 
The system captures video feed from a webcam or CCTV and performs real-time face detection using the MTCNN (Multi-task Cascaded Convolutional Networks) model.
Recognizes registered employees using pre-trained FaceNet models and marks their attendance automatically.
Automated Attendance Management: 
Automatically updates attendance records in a CSV file upon recognizing an employee's face. 
Maintains a daily attendance log with timestamps, reducing the need for manual tracking.
New User Registration:
Allows easy addition of new employees to the system. Users can register by capturing multiple images of their face through the web interface.
Generates unique face embeddings for each new user and stores them in the system.
Tailgating Detection:
Detects multiple faces in a single frame to identify potential tailgating (unauthorized access).
Saves images of the incident for security review and further action.
Alerts for Unrecognized Faces:
Identifies and tracks unknown or unregistered individuals attempting to gain access.
Automatically saves images of unrecognized faces in a designated folder for review.
User-Friendly Web Interface:
Provides a clean and interactive web interface for administrators to manage attendance, register new users, and monitor security alerts.
Security Alerts and Monitoring:
Maintains a log of security incidents, including unrecognized faces and tailgating attempts.
