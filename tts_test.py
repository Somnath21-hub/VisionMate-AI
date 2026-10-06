import cv2
import easyocr
import pyttsx3
import time
import queue
import threading

print("Loading OCR...")

reader = easyocr.Reader(['en'], gpu=True)

print("Loading speech engine...")

speech_queue = queue.Queue()


def speech_worker():

    engine = pyttsx3.init()

    engine.setProperty("rate", 150)
    engine.setProperty("volume", 1.0)

    while True:

        text = speech_queue.get()

        if text is None:
            break

        print("🔊 SPEAKING:", text)

        engine.say(text)
        engine.runAndWait()

        speech_queue.task_done()


# Start speech worker
speech_thread = threading.Thread(
    target=speech_worker,
    daemon=True
)

speech_thread.start()

print("VisionMate OCR + TTS ready!")

cap = cv2.VideoCapture(0)

last_ocr_time = 0

OCR_INTERVAL = 5


while True:

    ret, frame = cap.read()

    if not ret:
        break

    current_time = time.time()

    if current_time - last_ocr_time >= OCR_INTERVAL:

        last_ocr_time = current_time

        print("\n--- Reading text ---")

        results = reader.readtext(frame)

        detected_text = []

        for bbox, text, confidence in results:

            if confidence < 0.70:
                continue

            text = text.strip()

            if text == "":
                continue

            # Remove duplicate text
            if text.lower() not in [
                x.lower() for x in detected_text
            ]:
                detected_text.append(text)

        if detected_text:

            speech_text = ", ".join(detected_text)

            print("Detected:", speech_text)

            # Put speech into queue
            speech_queue.put(speech_text)

        else:

            print("No clear text detected.")

    cv2.imshow(
        "VisionMate OCR + TTS",
        frame
    )

    # ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break


# Stop everything
speech_queue.put(None)

cap.release()
cv2.destroyAllWindows()