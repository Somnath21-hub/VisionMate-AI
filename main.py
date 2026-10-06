from ultralytics import YOLO
from transformers import pipeline
import cv2
from PIL import Image
import numpy as np


# ============================================================
# 1. LOAD YOLO
# ============================================================

print("Loading YOLO...")

model = YOLO("yolo11n.pt")


# ============================================================
# 2. LOAD DEPTH MODEL
# ============================================================

print("Loading Depth Model on GPU...")

depth_model = pipeline(
    "depth-estimation",
    model="Intel/dpt-hybrid-midas",
    device=0
)

print("GPU: NVIDIA GeForce RTX 2050")
print("VisionMate started!")


# ============================================================
# 3. CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# ============================================================
# 4. VARIABLES
# ============================================================

# Previous bounding-box area
previous_area = {}


# Previous distance category
previous_distance = {}


# Number of frames for distance stabilization
distance_count = {}


# Reference depth for each tracked object
reference_depth = {}


# Reference bounding-box area
reference_area = {}


# Depth map
depth_map = None


# Frame counter
frame_count = 0


# Run depth model every 5 frames
DEPTH_INTERVAL = 5


# Distance must remain changed for 3 readings
STABLE_COUNT = 3


# ============================================================
# 5. DISTANCE FUNCTION
# ============================================================

def get_distance_level(track_id, depth, area):

    # --------------------------------------------------------
    # First time seeing this object
    # --------------------------------------------------------

    if track_id not in reference_depth:

        reference_depth[track_id] = depth
        reference_area[track_id] = area

        return "MEDIUM"


    # Get reference values

    ref_depth = reference_depth[track_id]
    ref_area = reference_area[track_id]


    # --------------------------------------------------------
    # Compare current depth with reference depth
    # --------------------------------------------------------

    depth_ratio = depth / ref_depth


    # --------------------------------------------------------
    # Compare current object size with reference size
    # --------------------------------------------------------

    area_ratio = area / ref_area


    # --------------------------------------------------------
    # NEAR
    # --------------------------------------------------------

    if depth_ratio < 0.75 or area_ratio > 1.50:

        return "NEAR"


    # --------------------------------------------------------
    # FAR
    # --------------------------------------------------------

    elif depth_ratio > 1.30 or area_ratio < 0.65:

        return "FAR"


    # --------------------------------------------------------
    # MEDIUM
    # --------------------------------------------------------

    else:

        return "MEDIUM"


# ============================================================
# 6. STABILIZE DISTANCE
# ============================================================

def stabilize_distance(track_id, new_distance):

    # --------------------------------------------------------
    # First reading
    # --------------------------------------------------------

    if track_id not in previous_distance:

        previous_distance[track_id] = new_distance
        distance_count[track_id] = 0

        return new_distance


    old_distance = previous_distance[track_id]


    # --------------------------------------------------------
    # Same distance
    # --------------------------------------------------------

    if new_distance == old_distance:

        distance_count[track_id] = 0

        return old_distance


    # --------------------------------------------------------
    # Distance changed
    # --------------------------------------------------------

    distance_count[track_id] += 1


    # --------------------------------------------------------
    # Change only after several consecutive readings
    # --------------------------------------------------------

    if distance_count[track_id] >= STABLE_COUNT:

        previous_distance[track_id] = new_distance

        distance_count[track_id] = 0

        return new_distance


    # Keep old distance until stable

    return old_distance


# ============================================================
# 7. MAIN LOOP
# ============================================================

while True:

    # --------------------------------------------------------
    # Read camera
    # --------------------------------------------------------

    ret, frame = cap.read()


    if not ret:

        print("Camera error!")

        break


    frame_count += 1


    height, width = frame.shape[:2]


    # ========================================================
    # 8. YOLO TRACKING
    # ========================================================

    results = model.track(
        frame,
        persist=True,
        device=0,
        verbose=False
    )


    result = results[0]


    # ========================================================
    # 9. DEPTH ESTIMATION
    # ========================================================

    if frame_count % DEPTH_INTERVAL == 0:

        # Convert BGR → RGB

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # Convert NumPy → PIL

        image = Image.fromarray(rgb)


        # Run depth model

        depth_result = depth_model(image)


        # Get depth map

        depth_map = np.array(
            depth_result["predicted_depth"]
        )


    # ========================================================
    # 10. PROCESS DETECTED OBJECTS
    # ========================================================

    if result.boxes is not None:

        for box in result.boxes:

            # ------------------------------------------------
            # Skip objects without tracking ID
            # ------------------------------------------------

            if box.id is None:

                continue


            # ------------------------------------------------
            # TRACK ID
            # ------------------------------------------------

            track_id = int(box.id[0])


            # ------------------------------------------------
            # BOUNDING BOX
            # ------------------------------------------------

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )


            # ------------------------------------------------
            # OBJECT CLASS
            # ------------------------------------------------

            class_id = int(box.cls[0])

            object_name = model.names[class_id]


            # =================================================
            # 11. LEFT / CENTER / RIGHT
            # =================================================

            center_x = (x1 + x2) // 2


            if center_x < width / 3:

                position = "LEFT"


            elif center_x < 2 * width / 3:

                position = "CENTER"


            else:

                position = "RIGHT"


            # =================================================
            # 12. OBJECT AREA
            # =================================================

            object_width = x2 - x1

            object_height = y2 - y1

            area = object_width * object_height


            # =================================================
            # 13. MOVEMENT
            # =================================================

            movement = "NEW"


            if track_id in previous_area:

                old_area = previous_area[track_id]


                # Object becoming larger

                if area > old_area * 1.10:

                    movement = "APPROACHING"


                # Object becoming smaller

                elif area < old_area * 0.90:

                    movement = "MOVING AWAY"


                # Almost same size

                else:

                    movement = "STABLE"


            # Save current area

            previous_area[track_id] = area


            # =================================================
            # 14. DEPTH
            # =================================================

            depth_value = None

            distance = "UNKNOWN"


            if depth_map is not None:

                depth_height, depth_width = depth_map.shape[:2]


                # Convert YOLO coordinates
                # to depth-map coordinates

                dx1 = int(
                    x1 / width * depth_width
                )

                dy1 = int(
                    y1 / height * depth_height
                )

                dx2 = int(
                    x2 / width * depth_width
                )

                dy2 = int(
                    y2 / height * depth_height
                )


                # Keep coordinates inside depth map

                dx1 = max(
                    0,
                    min(dx1, depth_width - 1)
                )

                dy1 = max(
                    0,
                    min(dy1, depth_height - 1)
                )

                dx2 = max(
                    0,
                    min(dx2, depth_width)
                )

                dy2 = max(
                    0,
                    min(dy2, depth_height)
                )


                # Get object depth region

                object_depth = depth_map[
                    dy1:dy2,
                    dx1:dx2
                ]


                if object_depth.size > 0:

                    # Median is more stable than mean

                    depth_value = float(
                        np.median(object_depth)
                    )


                    # =================================================
                    # 15. DISTANCE
                    # =================================================

                    raw_distance = get_distance_level(
                        track_id,
                        depth_value,
                        area
                    )


                    # Stabilize distance

                    distance = stabilize_distance(
                        track_id,
                        raw_distance
                    )


            # =================================================
            # 16. PRINT RESULT
            # =================================================

            print(
                "ID:", track_id,
                "| Object:", object_name,
                "| Position:", position,
                "| Movement:", movement,
                "| Depth:",
                round(depth_value, 2)
                if depth_value is not None
                else "N/A",
                "| Distance:", distance
            )


    # ========================================================
    # 17. DISPLAY
    # ========================================================

    output = result.plot()


    cv2.imshow(
        "VisionMate - GPU",
        output
    )


    # ========================================================
    # 18. EXIT
    # ========================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ============================================================
# 19. CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()