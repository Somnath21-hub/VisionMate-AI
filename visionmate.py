import cv2
import time
import numpy as np
import pyttsx3
import easyocr

from ultralytics import YOLO
from PIL import Image
from transformers import pipeline


# ============================================================
# 1. LOAD MODELS
# ============================================================

print("Loading YOLO...")
model = YOLO("yolo11n.pt")

print("Loading Depth Model...")
depth_model = pipeline(
    "depth-estimation",
    model="Intel/dpt-hybrid-midas",
    device=0
)

print("Loading OCR...")
reader = easyocr.Reader(
    ['en'],
    gpu=True
)

print("Loading Speech Engine...")
engine = pyttsx3.init()
engine.setProperty("rate", 150)
engine.setProperty("volume", 1.0)

print("VisionMate Ready!")


# ============================================================
# 2. CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()


# ============================================================
# 3. TRACKING VARIABLES
# ============================================================

previous_area = {}
last_alert_time = {}

ALERT_COOLDOWN = 5


# ============================================================
# 4. OCR VARIABLES
# ============================================================

last_ocr_time = 0
OCR_INTERVAL = 5


# ============================================================
# 5. SPEAK FUNCTION
# ============================================================

def speak(text):

    print("🔊 SPEAKING:", text)

    engine.say(text)
    engine.runAndWait()


# ============================================================
# 6. POSITION FUNCTION
# ============================================================

def get_position(center_x, width):

    if center_x < width / 3:

        return "LEFT"

    elif center_x < 2 * width / 3:

        return "CENTER"

    else:

        return "RIGHT"


# ============================================================
# 7. MOVEMENT FUNCTION
# ============================================================

def get_movement(track_id, area):

    if track_id not in previous_area:

        previous_area[track_id] = area

        return "NEW"

    old_area = previous_area[track_id]

    previous_area[track_id] = area

    if area > old_area * 1.10:

        return "APPROACHING"

    elif area < old_area * 0.90:

        return "MOVING AWAY"

    else:

        return "STABLE"


# ============================================================
# 8. CONTEXT / RISK ENGINE
# ============================================================

def calculate_priority(
    object_name,
    position,
    movement,
    depth_value
):

    score = 0

    # Important objects
    important_objects = [
        "person",
        "car",
        "bus",
        "truck",
        "motorcycle",
        "bicycle",
        "train"
    ]

    if object_name in important_objects:

        score += 3

    else:

        score += 1


    # Movement
    if movement == "APPROACHING":

        score += 3

    elif movement == "MOVING AWAY":

        score += 1


    # Position
    if position == "CENTER":

        score += 2

    else:

        score += 1


    # Relative depth
    if depth_value is not None:

        if depth_value < 1200:

            score += 3

        elif depth_value < 1600:

            score += 2

        else:

            score += 1


    # Priority
    if score >= 8:

        return "HIGH"

    elif score >= 5:

        return "MEDIUM"

    else:

        return "LOW"


# ============================================================
# 9. GENERATE INTELLIGENT ALERT
# ============================================================

def generate_alert(
    object_name,
    position,
    movement,
    priority
):

    # Only HIGH priority objects generate alerts
    if priority != "HIGH":

        return None


    # Approaching object
    if movement == "APPROACHING":

        if position == "CENTER":

            return (
                f"{object_name} "
                f"approaching in front of you"
            )

        else:

            return (
                f"{object_name} "
                f"approaching from your "
                f"{position.lower()}"
            )


    # Object in front
    elif position == "CENTER":

        return (
            f"{object_name} "
            f"in front of you"
        )


    # Object on side
    else:

        return (
            f"{object_name} "
            f"on your "
            f"{position.lower()}"
        )


# ============================================================
# 10. MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:

        print("Could not read camera frame.")

        break


    height, width = frame.shape[:2]


    # ========================================================
    # YOLO TRACKING
    # ========================================================

    results = model.track(
        frame,
        persist=True,
        verbose=False
    )


    # ========================================================
    # DEPTH ESTIMATION
    # ========================================================

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    image = Image.fromarray(rgb)

    depth_result = depth_model(image)

    depth_map = np.array(
        depth_result["predicted_depth"]
    )


    # Resize depth map
    depth_map = cv2.resize(
        depth_map,
        (width, height)
    )


    # ========================================================
    # PROCESS DETECTIONS
    # ========================================================

    if results[0].boxes is not None:

        boxes = results[0].boxes


        for i in range(len(boxes)):

            # ------------------------------------------------
            # Bounding Box
            # ------------------------------------------------

            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)


            # ------------------------------------------------
            # Confidence
            # ------------------------------------------------

            confidence = float(
                boxes.conf[i].cpu().numpy()
            )


            if confidence < 0.40:

                continue


            # ------------------------------------------------
            # Class
            # ------------------------------------------------

            class_id = int(
                boxes.cls[i].cpu().numpy()
            )

            object_name = model.names[class_id]


            # ------------------------------------------------
            # Track ID
            # ------------------------------------------------

            if boxes.id is not None:

                track_id = int(
                    boxes.id[i].cpu().numpy()
                )

            else:

                track_id = i


            # ------------------------------------------------
            # Center
            # ------------------------------------------------

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2


            # ------------------------------------------------
            # Position
            # ------------------------------------------------

            position = get_position(
                center_x,
                width
            )


            # ------------------------------------------------
            # Area
            # ------------------------------------------------

            area = max(
                1,
                (x2 - x1) * (y2 - y1)
            )


            # ------------------------------------------------
            # Movement
            # ------------------------------------------------

            movement = get_movement(
                track_id,
                area
            )


            # ------------------------------------------------
            # Depth
            # ------------------------------------------------

            object_depth = depth_map[
                max(0, center_y),
                max(0, center_x)
            ]


            # ------------------------------------------------
            # Priority
            # ------------------------------------------------

            priority = calculate_priority(
                object_name,
                position,
                movement,
                object_depth
            )


            # =================================================
            # PRINT INFORMATION
            # =================================================

            print(
                f"ID: {track_id} | "
                f"Object: {object_name} | "
                f"Position: {position} | "
                f"Movement: {movement} | "
                f"Depth: {object_depth:.2f} | "
                f"Priority: {priority}"
            )


            # =================================================
            # INTELLIGENT VOICE ALERT
            # =================================================

            alert = generate_alert(
                object_name,
                position,
                movement,
                priority
            )


            if alert is not None:

                current_time = time.time()

                last_time = last_alert_time.get(
                    track_id,
                    0
                )


                # 5-second cooldown
                if (
                    current_time - last_time
                    >= ALERT_COOLDOWN
                ):

                    speak(alert)

                    last_alert_time[
                        track_id
                    ] = current_time


            # =================================================
            # DRAW BOUNDING BOX
            # =================================================

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            # =================================================
            # DISPLAY LABEL
            # =================================================

            label = (
                f"{object_name} "
                f"{confidence:.2f}"
            )

            cv2.putText(
                frame,
                label,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )


            # =================================================
            # DISPLAY POSITION
            # =================================================

            cv2.putText(
                frame,
                f"{position} | {movement}",
                (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                2
            )


            # =================================================
            # DISPLAY PRIORITY
            # =================================================

            cv2.putText(
                frame,
                f"Priority: {priority}",
                (x1, y2 + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )


    # ========================================================
    # OCR EVERY 5 SECONDS
    # ========================================================

    current_time = time.time()


    if current_time - last_ocr_time >= OCR_INTERVAL:

        last_ocr_time = current_time

        print("\n--- Reading text ---")

        ocr_results = reader.readtext(frame)

        detected_text = []


        for bbox, text, confidence in ocr_results:

            if confidence < 0.70:

                continue

            detected_text.append(text)


        if detected_text:

            final_text = ", ".join(
                detected_text
            )

            print(
                "Detected:",
                final_text
            )

            speak(final_text)

        else:

            print(
                "No clear text detected."
            )


    # ========================================================
    # SHOW CAMERA
    # ========================================================

    cv2.imshow(
        "VisionMate - AI Visual Assistant",
        frame
    )


    # ========================================================
    # ESC TO EXIT
    # ========================================================

    if cv2.waitKey(1) & 0xFF == 27:

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

engine.stop()

print("VisionMate stopped.")