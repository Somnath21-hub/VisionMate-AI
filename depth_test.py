import cv2
import torch
from PIL import Image
from transformers import pipeline
import numpy as np

# Load depth model
depth_model = pipeline(
    "depth-estimation",
    model="Intel/dpt-hybrid-midas"
)

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # OpenCV BGR -> RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # NumPy -> PIL Image
    image = Image.fromarray(rgb)

    # Depth estimation
    result = depth_model(image)

    # Get depth image
    depth = np.array(result["depth"])

    # Normalize for display
    depth = cv2.normalize(
        depth,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    # Resize
    depth = cv2.resize(
        depth,
        (frame.shape[1], frame.shape[0])
    )

    cv2.imshow("Camera", frame)
    cv2.imshow("Depth", depth)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()