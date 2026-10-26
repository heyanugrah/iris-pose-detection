import os
import cv2
import torch
import torch.nn as nn
import numpy as np

from PIL import Image
from torchvision import models, transforms


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_PATH = (
    r"C:\Users\chethana\Desktop\sk.jpg"
)

MODEL_PATH = (
    r"C:\Users\chethana"
    r"\PycharmProjects"
    r"\RealtimeVideoAnalytics"
    r"\poseidentification"
    r"\best_eye_pose_model.pth"
)

OUTPUT_IMAGE = (
    r"C:\Users\chethana"
    r"\PycharmProjects"
    r"\RealtimeVideoAnalytics"
    r"\poseidentification"
    r"\prediction.jpg"
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 224

HEATMAP_SIZE = 28

NUM_KEYPOINTS = 11

WINDOW_RADIUS = 1

SOFTMAX_TEMPERATURE = 0.10

WINDOW_WIDTH = 900
WINDOW_HEIGHT = 700

MAX_ZOOM = 10.0


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print()
print("=" * 70)
print("DEVICE")
print("=" * 70)

print("Device:", device)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# KEYPOINT NAMES
# ============================================================

KEYPOINT_NAMES = [

    # LEFT EYE
    "left_eye_1",
    "left_eye_2",
    "left_eye_3",
    "left_eye_4",
    "left_iris_center",

    # RIGHT EYE
    "right_eye_1",
    "right_eye_2",
    "right_eye_3",
    "right_eye_4",
    "right_iris_center",

    # NOSE
    "nose_center"
]


# ============================================================
# SKELETON
# ============================================================

SKELETON = [

    # LEFT EYE
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),

    # LEFT EYE -> IRIS CENTER
    (0, 4),
    (1, 4),
    (2, 4),
    (3, 4),

    # RIGHT EYE
    (5, 6),
    (6, 7),
    (7, 8),
    (8, 5),

    # RIGHT EYE -> IRIS CENTER
    (5, 9),
    (6, 9),
    (7, 9),
    (8, 9),

    # IRIS -> NOSE
    (4, 10),
    (9, 10)
]


# ============================================================
# MODEL
# ============================================================

class EyePoseModel(nn.Module):

    def __init__(self):

        super().__init__()

        # ====================================================
        # RESNET-18
        # ====================================================

        resnet = models.resnet18(
            weights=None
        )

        # ====================================================
        # FEATURE EXTRACTOR
        # ====================================================

        self.features = nn.Sequential(

            resnet.conv1,

            resnet.bn1,

            resnet.relu,

            resnet.maxpool,

            resnet.layer1,

            resnet.layer2,

            resnet.layer3,

            resnet.layer4
        )

        # ====================================================
        # UPSAMPLING
        # ====================================================

        self.upsample = nn.Sequential(

            nn.ConvTranspose2d(

                512,

                256,

                kernel_size=4,

                stride=2,

                padding=1
            ),

            nn.BatchNorm2d(
                256
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.ConvTranspose2d(

                256,

                128,

                kernel_size=4,

                stride=2,

                padding=1
            ),

            nn.BatchNorm2d(
                128
            ),

            nn.ReLU(
                inplace=True
            )
        )

        # ====================================================
        # POSE HEAD
        # ====================================================

        self.pose_head = nn.Conv2d(

            128,

            NUM_KEYPOINTS,

            kernel_size=1
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        x
    ):

        x = self.features(x)

        x = self.upsample(x)

        x = self.pose_head(x)

        return x


# ============================================================
# CREATE MODEL
# ============================================================

print()
print("=" * 70)
print("CREATING MODEL")
print("=" * 70)

model = EyePoseModel()


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print()
print("=" * 70)
print("LOADING CHECKPOINT")
print("=" * 70)

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(

        "Model checkpoint not found:\n"
        + MODEL_PATH
    )


checkpoint = torch.load(

    MODEL_PATH,

    map_location=device
)


# ============================================================
# CHECKPOINT FORMAT
# ============================================================

if isinstance(
    checkpoint,
    dict
):

    if "model_state_dict" in checkpoint:

        state_dict = (
            checkpoint["model_state_dict"]
        )

    elif "state_dict" in checkpoint:

        state_dict = (
            checkpoint["state_dict"]
        )

    else:

        state_dict = checkpoint

else:

    state_dict = checkpoint


# ============================================================
# REMOVE module. PREFIX
# ============================================================

clean_state_dict = {}


for key, value in state_dict.items():

    if key.startswith("module."):

        key = key[
            len("module.") :
        ]

    clean_state_dict[key] = value


# ============================================================
# LOAD MODEL WEIGHTS
# ============================================================

try:

    model.load_state_dict(

        clean_state_dict,

        strict=True
    )

except RuntimeError as e:

    print()
    print("=" * 70)
    print("MODEL ARCHITECTURE MISMATCH")
    print("=" * 70)

    print(e)

    raise


# ============================================================
# MOVE MODEL TO DEVICE
# ============================================================

model = model.to(
    device
)


# ============================================================
# EVALUATION MODE
# ============================================================

model.eval()


print(
    "Model loaded successfully."
)


# ============================================================
# VERIFY MODEL OUTPUT
# ============================================================

print()
print("=" * 70)
print("VERIFYING MODEL")
print("=" * 70)


with torch.no_grad():

    dummy_input = torch.randn(

        1,

        3,

        IMAGE_SIZE,

        IMAGE_SIZE

    ).to(device)

    dummy_output = model(
        dummy_input
    )


expected_shape = (

    1,

    NUM_KEYPOINTS,

    HEATMAP_SIZE,

    HEATMAP_SIZE
)


print(
    "Model output:",
    tuple(dummy_output.shape)
)

print(
    "Expected:",
    expected_shape
)


if tuple(dummy_output.shape) != expected_shape:

    raise RuntimeError(

        "\nMODEL OUTPUT MISMATCH\n"

        f"Expected: {expected_shape}\n"

        f"Got: {tuple(dummy_output.shape)}"
    )


print(
    "Model architecture verified."
)


# ============================================================
# TRANSFORMATION
# ============================================================

transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[

            0.485,
            0.456,
            0.406

        ],

        std=[

            0.229,
            0.224,
            0.225

        ]
    )
])


# ============================================================
# LOAD IMAGE
# ============================================================

print()
print("=" * 70)
print("LOADING IMAGE")
print("=" * 70)

print(
    "Image:",
    IMAGE_PATH
)


if not os.path.exists(IMAGE_PATH):

    raise FileNotFoundError(

        "Input image not found:\n"
        + IMAGE_PATH
    )


image = cv2.imread(
    IMAGE_PATH
)


if image is None:

    raise FileNotFoundError(

        "Could not read image:\n"
        + IMAGE_PATH
    )


# ============================================================
# ORIGINAL IMAGE
# ============================================================

original_image = image.copy()


original_height, original_width = (
    image.shape[:2]
)


print(
    "Original image size:",
    original_width,
    "x",
    original_height
)


# ============================================================
# BGR -> RGB -> PIL
# ============================================================

rgb_image = cv2.cvtColor(

    image,

    cv2.COLOR_BGR2RGB
)

pil_image = Image.fromarray(
    rgb_image
)


# ============================================================
# RESIZE -> TRANSFORM
# ============================================================

resized_image = pil_image.resize(

    (
        IMAGE_SIZE,
        IMAGE_SIZE
    )
)

input_tensor = transform(
    resized_image
)


input_tensor = input_tensor.unsqueeze(
    0
)


input_tensor = input_tensor.to(
    device
)


print(
    "Input tensor:",
    tuple(input_tensor.shape)
)


# ============================================================
# PREDICTION
# ============================================================

print()
print("=" * 70)
print("RUNNING PREDICTION")
print("=" * 70)


with torch.no_grad():

    output = model(
        input_tensor
    )


print(
    "Output shape:",
    tuple(output.shape)
)


if tuple(output.shape) != expected_shape:

    raise RuntimeError(

        "\nWRONG OUTPUT SHAPE\n"

        f"Expected: {expected_shape}\n"

        f"Got: {tuple(output.shape)}"
    )


# ============================================================
# REMOVE BATCH DIMENSION
# ============================================================

heatmaps = output[0].detach().cpu()


HEATMAP_SCALE = (

    IMAGE_SIZE
    /
    HEATMAP_SIZE

)


print(
    "Heatmap scale:",
    HEATMAP_SCALE
)


# ============================================================
# PREDICT KEYPOINTS
# ============================================================

predicted_keypoints = []


for index in range(
    NUM_KEYPOINTS
):

    heatmap = heatmaps[index]

    flat_index = torch.argmax(
        heatmap
    )

    max_y = (
        flat_index
        //
        HEATMAP_SIZE
    ).item()

    max_x = (
        flat_index
        %
        HEATMAP_SIZE
    ).item()

    max_value = heatmap[
        max_y,
        max_x
    ].item()

    x_start = max(
        0,
        max_x - WINDOW_RADIUS
    )

    x_end = min(
        HEATMAP_SIZE,
        max_x + WINDOW_RADIUS + 1
    )

    y_start = max(
        0,
        max_y - WINDOW_RADIUS
    )

    y_end = min(
        HEATMAP_SIZE,
        max_y + WINDOW_RADIUS + 1
    )

    local_heatmap = heatmap[
        y_start:y_end,
        x_start:x_end
    ]

    values = (
        local_heatmap
        /
        SOFTMAX_TEMPERATURE
    )

    weights = torch.softmax(
        values.reshape(-1),
        dim=0
    )

    yy, xx = torch.meshgrid(

        torch.arange(
            y_start,
            y_end,
            dtype=torch.float32
        ),

        torch.arange(
            x_start,
            x_end,
            dtype=torch.float32
        ),

        indexing="ij"
    )

    xx = xx.reshape(-1)

    yy = yy.reshape(-1)

    refined_x = (
        xx * weights
    ).sum().item()

    refined_y = (
        yy * weights
    ).sum().item()

    x_224 = (
        refined_x
        *
        HEATMAP_SCALE
    )

    y_224 = (
        refined_y
        *
        HEATMAP_SCALE
    )

    original_x = (
        x_224
        *
        original_width
        /
        IMAGE_SIZE
    )

    original_y = (
        y_224
        *
        original_height
        /
        IMAGE_SIZE
    )

    original_x = int(
        round(original_x)
    )

    original_y = int(
        round(original_y)
    )

    original_x = max(
        0,
        min(
            original_x,
            original_width - 1
        )
    )

    original_y = max(
        0,
        min(
            original_y,
            original_height - 1
        )
    )

    predicted_keypoints.append({

        "id": index,

        "name":
            KEYPOINT_NAMES[index],

        "x":
            original_x,

        "y":
            original_y,

        "peak":
            max_value

    })


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 90)
print("PREDICTED KEYPOINTS")
print("=" * 90)


for point in predicted_keypoints:

    print(

        f"{point['id']:2d} | "

        f"{point['name']:20s} | "

        f"X = {point['x']:4d} | "

        f"Y = {point['y']:4d} | "

        f"Peak = {point['peak']:.6f}"

    )


# ============================================================
# CREATE DISPLAY IMAGE (MATCH ANNOTATION TOOL CANVAS)
# ============================================================

scale_x = WINDOW_WIDTH / original_width

scale_y = WINDOW_HEIGHT / original_height

base_scale = min(
    scale_x,
    scale_y
)

base_w = int(
    original_width * base_scale
)

base_h = int(
    original_height * base_scale
)

resized_original = cv2.resize(

    original_image,

    (
        base_w,
        base_h
    ),

    interpolation=cv2.INTER_AREA
)

canvas = np.zeros(

    (
        WINDOW_HEIGHT,
        WINDOW_WIDTH,
        3
    ),

    dtype=np.uint8
)

copy_w = min(
    base_w,
    WINDOW_WIDTH
)

copy_h = min(
    base_h,
    WINDOW_HEIGHT
)

canvas[
    0:copy_h,
    0:copy_w
] = resized_original[
    0:copy_h,
    0:copy_w
]

display_image = canvas.copy()


# ------------------------------------------------------------
# HELPER: ORIGINAL TO SCREEN
# ------------------------------------------------------------
def original_to_screen(x, y):

    return (
        int(x * base_scale),
        int(y * base_scale)
    )


# ============================================================
# DRAW SKELETON
# ============================================================

for a, b in SKELETON:

    if (
        a >= len(predicted_keypoints)
        or
        b >= len(predicted_keypoints)
    ):

        continue

    point_a = predicted_keypoints[a]

    point_b = predicted_keypoints[b]

    # Convert original to mapped screen coordinates
    x1, y1 = original_to_screen(
        point_a["x"],
        point_a["y"]
    )

    x2, y2 = original_to_screen(
        point_b["x"],
        point_b["y"]
    )


    # --------------------------------------------------------
    # SKELETON LINE
    # --------------------------------------------------------

    cv2.line(

        display_image,

        (x1, y1),

        (x2, y2),

        (255, 0, 0),

        2,

        cv2.LINE_AA

    )


# ============================================================
# DRAW KEYPOINTS
# ============================================================

for index, point in enumerate(
    predicted_keypoints
):

    # Convert original to mapped screen coordinates
    x, y = original_to_screen(
        point["x"],
        point["y"]
    )

    name = point["name"]


    # --------------------------------------------------------
    # ADJUSTED SMALLER POINT SIZE
    # --------------------------------------------------------

    if "iris_center" in name:

        radius = 3  # Reduced from 7

    elif name == "nose_center":

        radius = 3  # Reduced from 7

    else:

        radius = 2  # Reduced from 5


    # --------------------------------------------------------
    # RED POINT
    # --------------------------------------------------------

    cv2.circle(

        display_image,

        (x, y),

        radius,

        (0, 0, 255),

        -1,

        cv2.LINE_AA

    )


    # --------------------------------------------------------
    # INDEX
    # --------------------------------------------------------

    label = str(index)


    cv2.putText(

        display_image,

        label,

        (
            x + 6,
            y - 6
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.4,  # Slightly smaller font size

        (0, 0, 0),

        2,  # Thinner outline

        cv2.LINE_AA

    )


    cv2.putText(

        display_image,

        label,

        (
            x + 6,
            y - 6
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.4,  # Slightly smaller font size

        (255, 255, 255),

        1,

        cv2.LINE_AA

    )


# ============================================================
# SAVE RESULT
# ============================================================

output_directory = os.path.dirname(
    OUTPUT_IMAGE
)


if output_directory:

    os.makedirs(

        output_directory,

        exist_ok=True

    )


success = cv2.imwrite(

    OUTPUT_IMAGE,

    display_image

)


if not success:

    raise RuntimeError(

        "Could not save prediction image:\n"
        +
        OUTPUT_IMAGE

    )


# ============================================================
# PRINT SAVE LOCATION
# ============================================================

print()
print("=" * 70)
print("IMAGE SAVED")
print("=" * 70)

print(
    OUTPUT_IMAGE
)


# ============================================================
# DISPLAY
# ============================================================

window_name = (
    "Eye Pose Prediction"
)


cv2.namedWindow(

    window_name,

    cv2.WINDOW_AUTOSIZE

)


cv2.imshow(

    window_name,

    display_image

)


# ============================================================
# DONE
# ============================================================

print()
print("=" * 70)
print("DONE")
print("=" * 70)

print(
    "11 predicted points displayed."
)

print(
    "Press Q or ESC to close."
)


# ============================================================
# WAIT
# ============================================================

while True:

    key = cv2.waitKey(
        20
    ) & 0xFF


    if key == ord("q"):

        break


    if key == 27:

        break


# ============================================================
# CLOSE
# ============================================================

cv2.destroyAllWindows()