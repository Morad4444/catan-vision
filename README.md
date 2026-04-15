# Dice Detection With Logitech C270

This project detects up to three separate dice from a USB webcam stream and estimates the visible value on each die.

## What it does

- Captures live video from a webcam such as the Logitech C270 HD
- Searches for square contours that look like die faces
- Warps each die face into a top-down view
- Counts dark circular pips inside the die face
- Draws the detected dice and values on the video stream

## Assumptions

- The dice are standard light-colored dice with dark pips
- The three dice are visible and not heavily overlapping
- The background contrasts with the dice
- Lighting is reasonably even, without strong glare

## Setup

1. Create and activate a Python 3.11+ environment.
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Start the app:

```bash
python app.py
```

By default the app now tries to find a Windows camera whose device name matches `Logitech C270` and uses that device instead of the internal laptop webcam.

If your webcam is not the default camera, try another index:

```bash
python app.py --camera-index 1
```

If Windows exposes the Logitech device under a slightly different name, set it explicitly:

```bash
python app.py --camera-name "Logitech HD Webcam C270"
```

## Controls

- Press `q` to close the window

## Notes for better detection

- Place the dice on a matte surface with a different color than the dice
- Keep the camera above the table and avoid steep viewing angles
- Use stable lighting to reduce shadows and reflections
- Keep all three dice fully visible in the frame

## Next improvements

- Add HSV-based color segmentation for colored dice
- Calibrate pip size dynamically based on die size
- Add a debug window for threshold and contour inspection
- Save detections to a CSV or video file