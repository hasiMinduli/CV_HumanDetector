# Motion & Human Detection with Sobel + Color Thresholding 🎥🧍‍♂️

A fun computer vision project that uses:
- Sobel edge detection for motion tracking
- Skin-color thresholding for detecting humans
- Manual convolution, erosion, and dilation
- OpenCV for processing video frame-by-frame

## 🔧 How It Works
- Detects motion between frames using Sobel edge detection
- Identifies human-like regions using HSV skin color thresholding
- Highlights humans with green bounding boxes
- Displays motion regions with a binary difference mask

## 🚀 How to Run

1. Clone the repo
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
Replace the video path in main.py with your own:
```python
cap = cv2.VideoCapture("your_video.mp4")
```
Run it:

```bash
python main.py
```


🎥⚠️ **Privacy Note:** Due to privacy, the input video used for testing is not included. To test, use any short video with a person walking across the frame. Update the path to your video in `main.py`.

