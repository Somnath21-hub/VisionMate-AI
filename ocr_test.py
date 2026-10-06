import cv2
import easyocr
import time

print("Loading OCR model...")

reader = easyocr.Reader(['en'], gpu=True)

print("OCR ready!")

cap = cv2.VideoCapture(0)

last_ocr_time = 0
OCR_INTERVAL = 2

while True:

    ret, frame = cap.read()

    if not ret:
        break

    current_time = time.time()

    # Run OCR every 2 seconds
    if current_time - last_ocr_time >= OCR_INTERVAL:

        last_ocr_time = current_time

        print("\n--- Reading text ---")

        results = reader.readtext(frame)

        detected_text = []

        for bbox, text, confidence in results:

            # Ignore low-confidence results
            if confidence < 0.70:
                continue

            text = text.strip()

            if text == "":
                continue

            # Avoid duplicate text
            if text.lower() not in [
                x.lower() for x in detected_text
            ]:
                detected_text.append(text)

        if detected_text:

            for text in detected_text:
                print("Text:", text)

        else:
            print("No clear text detected.")

    cv2.imshow(
        "VisionMate OCR",
        frame
    )

    # ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()