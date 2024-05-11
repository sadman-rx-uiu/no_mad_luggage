import argparse
import time
import cv2
import mediapipe as mp
import RPi.GPIO as gpio
from picamera2 import Picamera2
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import bluetooth
import Adafruit_BBIO.GPIO as GPIO

# GPIO Pins for motor control
GPIO_FORWARD = 9
GPIO_BACKWARD = 13
GPIO_LEFT = 17
GPIO_RIGHT = 10

# Servo motor pin
SERVO_PIN = "P8_13"

# IR sensor pins
IR_SENSOR_PINS_FRONT = [18, 23, 24, 25]  # Pins for front sensors
IR_SENSOR_PINS_BACK = [8, 7, 12, 16]      # Pins for back sensors

# Bluetooth settings
UUID = "00001101-0000-1000-8000-00805F9B34FB"  # UUID for serial communication
SERVICE_NAME = "BluetoothService"
SERVER_PORT = 1

# Initialize GPIO
def init():
    gpio.setmode(gpio.BCM)
    gpio.setup(GPIO_FORWARD, gpio.OUT)
    gpio.setup(GPIO_BACKWARD, gpio.OUT)
    gpio.setup(GPIO_LEFT, gpio.OUT)
    gpio.setup(GPIO_RIGHT, gpio.OUT)
    GPIO.setup(SERVO_PIN, GPIO.OUT)
    for pin in IR_SENSOR_PINS_FRONT + IR_SENSOR_PINS_BACK:
        gpio.setup(pin, gpio.IN)

# Motor control functions
def forward():
    gpio.output(GPIO_FORWARD, True)
    gpio.output(GPIO_BACKWARD, False)
    gpio.output(GPIO_LEFT, True)
    gpio.output(GPIO_RIGHT, False)

def stop():
    gpio.output(GPIO_FORWARD, False)
    gpio.output(GPIO_BACKWARD, False)
    gpio.output(GPIO_LEFT, False)
    gpio.output(GPIO_RIGHT, False)
    
def left():
    gpio.output(GPIO_FORWARD, False)
    gpio.output(GPIO_BACKWARD, True)
    gpio.output(GPIO_LEFT, True)
    gpio.output(GPIO_RIGHT, False)
    
def right():
    gpio.output(GPIO_FORWARD, True)
    gpio.output(GPIO_BACKWARD, False)
    gpio.output(GPIO_LEFT, False)
    gpio.output(GPIO_RIGHT, True)

def rotate_servo():
    # Rotate servo motor by 90 degrees
    pwm = GPIO.PWM(SERVO_PIN, 50)
    pwm.start(7.5)
    pwm.ChangeDutyCycle(10)
    time.sleep(1)
    pwm.stop()

def read_weight_sensor():
    # Define the analog input pin
    analog_in = analogio.AnalogIn(board.A0)  # Replace A0 with the actual pin connected to the weight sensor

    # Read analog input value and convert it to weight
    analog_value = analog_in.value  # Read raw analog value
    # Convert analog value to weight (example: assuming linear relationship between analog value and weight)
    weight_grams = analog_value * 0.1  # Example conversion factor, adjust according to your sensor

    # Convert weight from grams to kilograms
    weight_kg = weight_grams / 1000.0  # Convert grams to kilograms

    return weight_kg

def wait_for_recognition_command():
    server_sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    server_sock.bind(("", SERVER_PORT))
    server_sock.listen(1)

    print("Waiting for Bluetooth connection...")
    client_sock, address = server_sock.accept()
    print("Accepted connection from", address)

    data = client_sock.recv(1024)
    client_sock.close()
    server_sock.close()

    return data.decode("utf-8")

def read_ir_sensors():
    sensor_values_front = [gpio.input(pin) for pin in IR_SENSOR_PINS_FRONT]
    sensor_values_back = [gpio.input(pin) for pin in IR_SENSOR_PINS_BACK]
    return sensor_values_front, sensor_values_back

def detect_obstacle(sensor_values_front, sensor_values_back):
    # Check if any front or back sensor detects an obstacle
    return any(sensor_values_front), any(sensor_values_back)

def main():
    # Wait for recognition command via Bluetooth
    print("Waiting for recognition command via Bluetooth...")
    while True:
        command = wait_for_recognition_command()
        if command.strip() in ["recognize", "weight"]:
            break
        else:
            print("Invalid command. Waiting for 'recognize' or 'weight' command...")

    # If the command is 'weight', send weight sensor data over Bluetooth
    if command.strip() == "weight":
        weight = read_weight_sensor()
        print("Weight in KG:", weight)
        server_sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
        server_sock.bind(("", SERVER_PORT))
        server_sock.listen(1)

        print("Waiting for Bluetooth connection...")
        client_sock, address = server_sock.accept()
        print("Accepted connection from", address)

        client_sock.send(str(weight))
        client_sock.close()
        server_sock.close()
        return

    # Parse command-line arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', help='Path to the gesture recognition model file.', required=False, default='/home/pi/Desktop/gesture2/mediapipe/examples/gesture_recognizer/raspberry_pi/gesture_recognizer.task')
    parser.add_argument('--numHands', help='Max number of hands that can be detected by the recognizer.', required=False, default=1)
    parser.add_argument('--minHandDetectionConfidence', help='The minimum confidence score for hand detection to be considered successful.', required=False, default=0.5)
    parser.add_argument('--minHandPresenceConfidence', help='The minimum confidence score of hand presence score in the hand landmark detection.', required=False, default=0.5)
    parser.add_argument('--minTrackingConfidence', help='The minimum confidence score for the hand tracking to be considered successful.', required=False, default=0.5)
    parser.add_argument('--cameraId', help='Id of camera.', required=False, default=0)
    parser.add_argument('--frameWidth', help='Width of frame to capture from camera.', required=False, default=640)
    parser.add_argument('--frameHeight', help='Height of frame to capture from camera.', required=False, default=480)
    args = parser.parse_args()

    # Initialize GPIO
    init()

    # Initialize Picamera2
    cv2.startWindowThread()
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"format": 'RGB888', "size": (640, 480)}))
    picam2.start()

    # Global variables for gesture recognition
    recognition_result_list = []

    def save_result(result: vision.GestureRecognizerResult, unused_output_image: mp.Image, timestamp_ms: int):
        recognition_result_list.append(result)

    # Initialize the gesture recognizer model
    base_options = python.BaseOptions(model_asset_path=args.model)
    options = vision.GestureRecognizerOptions(base_options=base_options,
                                          running_mode=vision.RunningMode.LIVE_STREAM,
                                          num_hands=int(args.numHands),
                                          min_hand_detection_confidence=float(args.minHandDetectionConfidence),
                                          min_hand_presence_confidence=float(args.minHandPresenceConfidence),
                                          min_tracking_confidence=float(args.minTrackingConfidence),
                                          result_callback=save_result)
    recognizer = vision.GestureRecognizer.create_from_options(options)

    # Initialize the face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    try:
        while True:
            # Capture frame from Picamera2
            frame = picam2.capture_array()

            # Convert BGR to RGB for MediaPipe
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

            # Detect face using OpenCV
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            # Read IR sensor values
            sensor_values_front, sensor_values_back = read_ir_sensors()

            # Check if face detected and no obstacle detected
            if len(faces) > 0 and not detect_obstacle(sensor_values_front, sensor_values_back):
                # Run gesture recognizer using the model
                recognizer.recognize_async(mp_image, time.time_ns() // 1_000_000)

                if recognition_result_list:
                    # Get the first gesture result
                    result = recognition_result_list[0]
                    
                    for hand_index, hand_landmarks in enumerate(
                        recognition_result_list[0].hand_landmarks):
                        # Calculate the bounding box of the hand
                        x_min = min([landmark.x for landmark in hand_landmarks])
                        y_min = min([landmark.y for landmark in hand_landmarks])
                        y_max = max([landmark.y for landmark in hand_landmarks])

                        # Convert normalized coordinates to pixel values
                        frame_height, frame_width = current_frame.shape[:2]
                        x_min_px = int(x_min * frame_width)
                        y_min_px = int(y_min * frame_height)
                        y_max_px = int(y_max * frame_height)
                        
                        # Get gesture classification results
                        if recognition_result_list[0].gestures:
                            gesture = recognition_result_list[0].gestures[hand_index]
                            category_name = gesture[0].category_name
                            score = round(gesture[0].score, 2)
                            result_text = f'{category_name} ({score})'

                            # Compute text size
                            text_size = \
                            cv2.getTextSize(result_text, cv2.FONT_HERSHEY_DUPLEX, label_font_size,
                                            label_thickness)[0]
                            text_width, text_height = text_size

                            # Calculate text position (above the hand)
                            text_x = x_min_px
                            text_y = y_min_px - 10  # Adjust this value as needed

                            # Make sure the text is within the frame boundaries
                            if text_y < 0:
                                text_y = y_max_px + text_height

                            # Draw the text
                            cv2.putText(current_frame, result_text, (text_x, text_y),
                                        cv2.FONT_HERSHEY_DUPLEX, label_font_size,
                                        label_text_color, label_thickness, cv2.LINE_AA)

                        # Draw hand landmarks on the frame
                        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                        hand_landmarks_proto.landmark.extend([
                        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y,
                                                        z=landmark.z) for landmark in
                        hand_landmarks
                        ])
                        mp_drawing.draw_landmarks(
                        current_frame,
                        hand_landmarks_proto,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                        

                    # Check if gestures are detected
                    if result.gestures:
                        for gesture in result.gestures:
                            category_name = gesture[0].category_name

                            # Call motor control functions based on gesture
                            if category_name == "Open_Palm":
                                forward()
                            elif category_name == "Closed_Fist":
                                stop()
                            else:
                                stop()
                            
                    recognition_result_list.clear()

            # Display frame
            cv2.imshow('Frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass

    finally:
        # Clean up GPIO and close Picamera2
        gpio.cleanup()
        picam2.stop()
        recognizer.close()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
