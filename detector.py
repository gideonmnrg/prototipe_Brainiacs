from keras.models import load_model
from collections import deque
import numpy as np
import cv2
import os

def print_results(video, limit=None):
    # Ensure output directory exists
    if not os.path.exists('output'):
        os.mkdir('output')

    print("Loading model ...")
    model = load_model('model3.h5')  # Load the trained model
    Q = deque(maxlen=128)
    vs = cv2.VideoCapture(video)
    writer = None
    (W, H) = (None, None)

    # Frame-related variables
    fps = int(vs.get(cv2.CAP_PROP_FPS))  # Get the frame rate of the video
    total_frames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0

    # Violence segment tracking
    is_violence = False
    start_frame = None
    violence_segments = []

    while True:
        # Read the next frame from the file
        (grabbed, frame) = vs.read()
        frame_count += 1

        # If the frame was not grabbed, we've reached the end
        if not grabbed:
            break

        # Grab frame dimensions if empty
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # Process the frame for the model
        output = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (128, 128)).astype("float32")
        frame = frame.reshape(128, 128, 3) / 255

        # Make predictions on the frame
        preds = model.predict(np.expand_dims(frame, axis=0))[0]
        Q.append(preds)

        # Prediction averaging
        results = np.array(Q).mean(axis=0)
        violence_detected = (results > 0.7)[0]  # Violence detection condition

        text_color = (0, 255, 0)  # Default: green

        # Track violence segments
        if violence_detected:
            text_color = (0, 0, 255)  # Red: violence detected
            if not is_violence:
                start_frame = frame_count  # Mark start of violence segment
                is_violence = True
        else:
            if is_violence:
                end_frame = frame_count  # Mark end of violence segment
                # Check duration of the segment in seconds
                duration = (end_frame - start_frame) / fps
                if duration > 1:  # Only save segments longer than 1 second
                    violence_segments.append((start_frame, end_frame))
                is_violence = False

        text = "Violence: {}".format(violence_detected)
        FONT = cv2.FONT_HERSHEY_SIMPLEX

        # Display the text on the frame
        cv2.putText(output, text, (35, 50), FONT, 1.25, text_color, 3)

        # Show the output frame
        cv2.imshow("Output", output)
        key = cv2.waitKey(1) & 0xFF

        # If 'q' key is pressed, exit the loop
        if key == ord("q"):
            break

    # Handle video segments saving after processing all frames
    if violence_segments:
        vs.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video to the beginning
        segment_count = 0

        for (start, end) in violence_segments:
            vs.set(cv2.CAP_PROP_POS_FRAMES, start)
            segment_writer = None
            print(f"[INFO] Saving violence segment {segment_count + 1}: frames {start} to {end}")
            
            for f in range(start, end):
                (grabbed, frame) = vs.read()
                if not grabbed:
                    break

                # Initialize writer for this segment if it's not created
                if segment_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                    output_path = f"output/violence_segment_{segment_count + 1}.mp4"
                    segment_writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H), True)

                # Write the frame to the segment output video
                segment_writer.write(frame)

            if segment_writer is not None:
                segment_writer.release()

            segment_count += 1

    # Release resources
    print("[INFO] cleaning up...")
    vs.release()
    cv2.destroyAllWindows()

# Modify the video path as per your local machine
V_path = "videoBegal.mp4"  # Local path to the violence video
NV_path = "nonv.mp4"  # Local path to the non-violence video

print_results(V_path)
