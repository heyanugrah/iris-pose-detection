# Iris Pose Detection

A PyTorch-based project for detecting 11 eye and face landmarks using a heatmap regression model. This repository trains a ResNet-18 pose estimator to locate the left eye, right eye, iris centers, and nose center from a single image.

## Overview

This project includes two main scripts:

- `pose.py` — loads the dataset, trains the model, and saves the best checkpoint
- `predict.py` — loads a trained model, predicts keypoints on a new image, and visualizes the result

The model predicts these 11 keypoints:

- left_eye_1
- left_eye_2
- left_eye_3
- left_eye_4
- left_iris_center
- right_eye_1
- right_eye_2
- right_eye_3
- right_eye_4
- right_iris_center
- nose_center

## Project Goal

The goal is to build a lightweight pose estimation model that can detect eye and iris landmarks in images and draw a compact skeleton around them.

## Model Architecture

The model uses a pretrained ResNet-18 backbone and converts the feature map into 11 keypoint heatmaps.

- Input size: 224 x 224 RGB image
- Backbone: ResNet-18
- Output: 11 heatmaps of size 28 x 28
- Training target: Gaussian heatmaps for each keypoint
- Loss: mean squared error (MSE)

The network follows this pattern:

- ResNet-18 feature extractor
- Two transpose-convolution upsampling stages
- 1x1 convolution pose head

## Dataset Format

The training code expects images and matching JSON annotation files to be placed in the same directory.

Example structure:

```text
dataset/
├── image_001.jpg
├── image_001.json
├── image_002.jpg
├── image_002.json
└── ...
```

Each JSON file contains a `keypoints` array with exactly 11 entries. Example:

```json
{
  "keypoints": [
    {"name": "left_eye_1", "x": 120, "y": 80, "visibility": 1.0},
    {"name": "left_eye_2", "x": 140, "y": 80, "visibility": 1.0},
    {"name": "left_eye_3", "x": 140, "y": 100, "visibility": 1.0},
    {"name": "left_eye_4", "x": 120, "y": 100, "visibility": 1.0},
    {"name": "left_iris_center", "x": 130, "y": 90, "visibility": 1.0},
    {"name": "right_eye_1", "x": 180, "y": 80, "visibility": 1.0},
    {"name": "right_eye_2", "x": 200, "y": 80, "visibility": 1.0},
    {"name": "right_eye_3", "x": 200, "y": 100, "visibility": 1.0},
    {"name": "right_eye_4", "x": 180, "y": 100, "visibility": 1.0},
    {"name": "right_iris_center", "x": 190, "y": 90, "visibility": 1.0},
    {"name": "nose_center", "x": 160, "y": 120, "visibility": 1.0}
  ]
}
```

## Training Pipeline

The `pose.py` script:

1. Finds all image files in the dataset folder
2. Sorts and shuffles them
3. Splits them into training and validation sets
4. Builds a PyTorch dataset that reads the image and corresponding annotation JSON
5. Scales points to the resized image dimensions
6. Generates Gaussian heatmaps for each keypoint
7. Trains the model using a ResNet-18 backbone
8. Saves the best model based on validation loss

Key configuration values in the script:

```python
IMAGE_SIZE = 224
HEATMAP_SIZE = 28
NUM_KEYPOINTS = 11
SIGMA = 1.5
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
EPOCHS = 50
TRAIN_SPLIT = 0.80
```

The best model is saved as:

```python
MODEL_SAVE_PATH = "best_eye_pose_model.pth"
```

## Prediction Pipeline

The `predict.py` script:

1. Loads the trained model checkpoint
2. Verifies the model architecture
3. Loads the input image
4. Resizes it to 224x224
5. Runs forward inference
6. Extracts the strongest response from each heatmap
7. Refines the point using a local weighted region
8. Maps the result back to the original image size
9. Draws the skeleton and keypoints
10. Saves the annotated output image

## Requirements

Install the required dependencies with:

```bash
pip install torch torchvision pillow opencv-python numpy
```

## How to Run

### Train the model

Update the dataset folder path inside `pose.py`, then run:

```bash
python pose.py
```

### Run predictions

Update the file paths in `predict.py`:

```python
IMAGE_PATH = r"C:\path\to\your\image.jpg"
MODEL_PATH = r"C:\path\to\best_eye_pose_model.pth"
OUTPUT_IMAGE = r"C:\path\to\prediction.jpg"
```

Then run:

```bash
python predict.py
```

The script will save the prediction image and display it in a window.

## Notes

- The project currently uses hardcoded local Windows paths, which should be replaced with relative paths or command-line arguments for portability.
- The dataset and annotation files are expected to be in the same folder.
- The model outputs heatmaps rather than direct coordinates, so the final keypoint positions are taken from the peak values of each heatmap.

## Example Output

The prediction script generates an image showing:

- red keypoint markers
- blue skeletal connections between landmarks
- index labels for each detected point

## Future Improvements

Possible improvements include:

- adding command-line argument support for paths
- supporting video/webcam inference
- improving coordinate refinement for better landmark accuracy
- adding better validation metrics
- creating a cleaner training pipeline with config files
- exporting the trained model for deployment

## License

This project does not currently include a license file. If you plan to share it publicly, consider adding an MIT or Apache 2.0 license.
