import os
import json
import random

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from torchvision.models import ResNet18_Weights, resnet18


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 224
HEATMAP_SIZE = 28
NUM_KEYPOINTS = 11

SIGMA = 1.5

BATCH_SIZE = 2
LEARNING_RATE = 1e-4
EPOCHS = 50

TRAIN_SPLIT = 0.80

RANDOM_SEED = 42

MODEL_SAVE_PATH = "best_eye_pose_model.pth"


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# RANDOM SEED
# ============================================================

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ============================================================
# KEYPOINT NAMES
# ============================================================

KEYPOINT_NAMES = [

    # Left eye
    "left_eye_1",
    "left_eye_2",
    "left_eye_3",
    "left_eye_4",
    "left_iris_center",

    # Right eye
    "right_eye_1",
    "right_eye_2",
    "right_eye_3",
    "right_eye_4",
    "right_iris_center",

    # Nose
    "nose_center"
]


# ============================================================
# SKELETON
# ============================================================

SKELETON = [

    # --------------------------------------------------------
    # Left eye
    # --------------------------------------------------------

    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),

    # --------------------------------------------------------
    # Left eye -> iris center
    # --------------------------------------------------------

    (0, 4),
    (1, 4),
    (2, 4),
    (3, 4),

    # --------------------------------------------------------
    # Right eye
    # --------------------------------------------------------

    (5, 6),
    (6, 7),
    (7, 8),
    (8, 5),

    # --------------------------------------------------------
    # Right eye -> iris center
    # --------------------------------------------------------

    (5, 9),
    (6, 9),
    (7, 9),
    (8, 9),

    # --------------------------------------------------------
    # Iris centers -> nose
    # --------------------------------------------------------

    (4, 10),
    (9, 10)
]


# ============================================================
# HEATMAP CREATION
# ============================================================

def heatmap_creation(all_keypoints):

    heatmaps = torch.zeros(
        NUM_KEYPOINTS,
        HEATMAP_SIZE,
        HEATMAP_SIZE,
        dtype=torch.float32
    )

    visibility = torch.zeros(
        NUM_KEYPOINTS,
        dtype=torch.float32
    )

    # --------------------------------------------------------
    # 224 -> 28
    # --------------------------------------------------------

    scale = IMAGE_SIZE / HEATMAP_SIZE

    # --------------------------------------------------------
    # Create heatmap for every keypoint
    # --------------------------------------------------------

    for idx, keypoint in all_keypoints.items():

        x = keypoint["x"]
        y = keypoint["y"]

        visibility[idx] = float(
            keypoint["visibility"]
        )

        # ----------------------------------------------------
        # Convert image coordinates
        # to heatmap coordinates
        # ----------------------------------------------------

        center_x = x / scale
        center_y = y / scale

        # ----------------------------------------------------
        # Generate Gaussian
        # ----------------------------------------------------

        for heatmap_y in range(HEATMAP_SIZE):

            for heatmap_x in range(HEATMAP_SIZE):

                distance = (
                    (heatmap_x - center_x) ** 2
                    +
                    (heatmap_y - center_y) ** 2
                )

                value = torch.exp(
                    torch.tensor(
                        -distance /
                        (2 * SIGMA * SIGMA),
                        dtype=torch.float32
                    )
                )

                heatmaps[
                    idx,
                    heatmap_y,
                    heatmap_x
                ] = value

    return heatmaps, visibility


# ============================================================
# DATASET
# ============================================================

class EyePoseDataset(TorchDataset):

    def __init__(
        self,
        image_folder,
        json_folder,
        image_files
    ):

        self.imageFolder = image_folder
        self.jsonFolder = json_folder

        # ----------------------------------------------------
        # These are the files assigned to this dataset
        # ----------------------------------------------------

        self.image_files = image_files

        # ----------------------------------------------------
        # ImageNet normalization
        # for pretrained ResNet-18
        # ----------------------------------------------------

        self.transform = transforms.Compose([

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


    # ========================================================
    # LENGTH
    # ========================================================

    def __len__(self):

        return len(self.image_files)


    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(self, index):

        input_image = self.image_files[index]

        image_path = os.path.join(
            self.imageFolder,
            input_image
        )

        # ----------------------------------------------------
        # LOAD IMAGE
        # ----------------------------------------------------

        image = Image.open(
            image_path
        ).convert("RGB")

        original_width, original_height = image.size

        # ----------------------------------------------------
        # RESIZE IMAGE
        # ----------------------------------------------------

        image = image.resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            )
        )

        # ----------------------------------------------------
        # Calculate coordinate scaling
        # ----------------------------------------------------

        x_scale = (
            IMAGE_SIZE /
            original_width
        )

        y_scale = (
            IMAGE_SIZE /
            original_height
        )

        # ----------------------------------------------------
        # JSON filename
        # ----------------------------------------------------

        json_name = (
            os.path.splitext(input_image)[0]
            + ".json"
        )

        json_path = os.path.join(
            self.jsonFolder,
            json_name
        )

        # ----------------------------------------------------
        # CHECK JSON
        # ----------------------------------------------------

        if not os.path.exists(json_path):

            raise FileNotFoundError(
                f"JSON file not found:\n{json_path}"
            )

        # ----------------------------------------------------
        # LOAD JSON
        # ----------------------------------------------------

        with open(
            json_path,
            "r"
        ) as file:

            data = json.load(file)

        json_keypoints = data["keypoints"]

        # ----------------------------------------------------
        # CHECK KEYPOINT COUNT
        # ----------------------------------------------------

        if len(json_keypoints) != NUM_KEYPOINTS:

            raise ValueError(
                f"\nImage: {input_image}"
                f"\nExpected: {NUM_KEYPOINTS} keypoints"
                f"\nFound: {len(json_keypoints)}"
            )

        converted_keypoints = {}

        # ----------------------------------------------------
        # SCALE KEYPOINTS
        # ----------------------------------------------------

        for idx, keypoint in enumerate(
            json_keypoints
        ):

            x_scaled = (
                float(keypoint["x"])
                * x_scale
            )

            y_scaled = (
                float(keypoint["y"])
                * y_scale
            )

            converted_keypoints[idx] = {

                "x": x_scaled,

                "y": y_scaled,

                "visibility": float(
                    keypoint["visibility"]
                ),

                "name": keypoint["name"]
            }

        # ----------------------------------------------------
        # IMAGE -> TENSOR
        # ----------------------------------------------------

        image_tensor = self.transform(
            image
        )

        # ----------------------------------------------------
        # CREATE HEATMAPS
        # ----------------------------------------------------

        heatmaps, visibility = heatmap_creation(
            converted_keypoints
        )

        return (
            image_tensor,
            heatmaps,
            visibility
        )


# ============================================================
# GET ALL IMAGES
# ============================================================

DATASET_FOLDER = (
    r"C:\Users\chethana"
    r"\PycharmProjects"
    r"\RealtimeVideoAnalytics"
    r"\dataset"
)


all_image_files = []

for file in os.listdir(
    DATASET_FOLDER
):

    if file.lower().endswith(
        (
            ".jpg",
            ".jpeg",
            ".png"
        )
    ):

        all_image_files.append(file)


# ============================================================
# SORT
# ============================================================

all_image_files.sort()


# ============================================================
# CHECK DATASET
# ============================================================

print()
print("=" * 60)

print(
    "Total images found:",
    len(all_image_files)
)

print("=" * 60)


# ============================================================
# SHUFFLE
# ============================================================

random.shuffle(
    all_image_files
)


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

split_index = int(
    len(all_image_files)
    * TRAIN_SPLIT
)


train_images = (
    all_image_files[
        :split_index
    ]
)


val_images = (
    all_image_files[
        split_index:
    ]
)


# ============================================================
# PRINT SPLIT
# ============================================================

print()
print(
    "Training images:",
    len(train_images)
)

print(
    "Validation images:",
    len(val_images)
)

print(
    "Training percentage:",
    f"{TRAIN_SPLIT * 100:.0f}%"
)

print(
    "Validation percentage:",
    f"{(1 - TRAIN_SPLIT) * 100:.0f}%"
)

print()


# ============================================================
# CHECK FOR DUPLICATES
# ============================================================

train_set = set(
    train_images
)

val_set = set(
    val_images
)

overlap = (
    train_set
    .intersection(val_set)
)


print(
    "Images appearing in both sets:",
    len(overlap)
)


if len(overlap) > 0:

    raise RuntimeError(
        "ERROR: Training and validation "
        "sets contain duplicate images!"
    )


# ============================================================
# CREATE TRAIN DATASET
# ============================================================

train_dt = EyePoseDataset(

    image_folder=DATASET_FOLDER,

    json_folder=DATASET_FOLDER,

    image_files=train_images
)


# ============================================================
# CREATE VALIDATION DATASET
# ============================================================

val_dt = EyePoseDataset(

    image_folder=DATASET_FOLDER,

    json_folder=DATASET_FOLDER,

    image_files=val_images
)


# ============================================================
# DATASET COUNTS
# ============================================================

print()
print("=" * 60)

print(
    "Train dataset:",
    len(train_dt)
)

print(
    "Validation dataset:",
    len(val_dt)
)

print("=" * 60)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(

    train_dt,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=0,

    pin_memory=torch.cuda.is_available()
)


val_loader = DataLoader(

    val_dt,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0,

    pin_memory=torch.cuda.is_available()
)


# ============================================================
# SANITY CHECK
# ============================================================

print()
print("=" * 60)

print("Running sanity check...")

print("=" * 60)


images, target_heatmaps, visibility = next(
    iter(train_loader)
)


print(
    "Images:",
    images.shape
)

print(
    "Target heatmaps:",
    target_heatmaps.shape
)

print(
    "Visibility:",
    visibility.shape
)


# ============================================================
# EXPECTED SHAPES
# ============================================================

print()
print("Expected shapes:")

print(
    "[B, 3, 224, 224]"
)

print(
    "[B, 11, 28, 28]"
)

print(
    "[B, 11]"
)

print()


# ============================================================
# MODEL
# ============================================================

class Model(nn.Module):

    def __init__(self):

        super().__init__()

        # ----------------------------------------------------
        # PRETRAINED RESNET-18
        # ----------------------------------------------------

        backbone = resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        # ----------------------------------------------------
        # FEATURE EXTRACTOR
        #
        # Input:
        # [B, 3, 224, 224]
        #
        # Output:
        # [B, 512, 7, 7]
        # ----------------------------------------------------

        self.features = nn.Sequential(

            backbone.conv1,

            backbone.bn1,

            backbone.relu,

            backbone.maxpool,

            backbone.layer1,

            backbone.layer2,

            backbone.layer3,

            backbone.layer4
        )

        # ----------------------------------------------------
        # UPSAMPLING
        #
        # 7x7 -> 14x14
        #
        # 14x14 -> 28x28
        # ----------------------------------------------------

        self.upsample = nn.Sequential(

            nn.ConvTranspose2d(

                512,

                256,

                kernel_size=4,

                stride=2,

                padding=1
            ),

            nn.BatchNorm2d(256),

            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(

                256,

                128,

                kernel_size=4,

                stride=2,

                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(inplace=True)
        )

        # ----------------------------------------------------
        # POSE HEAD
        #
        # 128 -> 11 heatmaps
        # ----------------------------------------------------

        self.pose_head = nn.Conv2d(

            128,

            NUM_KEYPOINTS,

            kernel_size=1
        )


    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x):

        # ----------------------------------------------------
        # ResNet
        # ----------------------------------------------------

        x = self.features(x)

        # ----------------------------------------------------
        # Upsample
        # ----------------------------------------------------

        x = self.upsample(x)

        # ----------------------------------------------------
        # Keypoint heatmaps
        # ----------------------------------------------------

        x = self.pose_head(x)

        return x


# ============================================================
# INITIALIZE MODEL
# ============================================================

my_model = Model()

my_model = my_model.to(device)


# ============================================================
# PRINT MODEL OUTPUT TEST
# ============================================================

with torch.no_grad():

    test_input = torch.randn(
        1,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE
    ).to(device)

    test_output = my_model(
        test_input
    )


print()
print("=" * 60)

print(
    "Model output shape:",
    test_output.shape
)

print(
    "Expected:",
    (
        "1,",
        NUM_KEYPOINTS,
        HEATMAP_SIZE,
        HEATMAP_SIZE
    )
)

print("=" * 60)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(

    my_model.parameters(),

    lr=LEARNING_RATE
)


# ============================================================
# BEST VALIDATION LOSS
# ============================================================

best_val_loss = float("inf")


# ============================================================
# TRAINING
# ============================================================

for epoch in range(EPOCHS):


    # ========================================================
    # TRAIN MODE
    # ========================================================

    my_model.train()

    total_loss = 0.0


    # ========================================================
    # TRAINING BATCHES
    # ========================================================

    for (
        batch_images,
        batch_target_heatmaps,
        batch_visibility
    ) in train_loader:


        # ----------------------------------------------------
        # MOVE TO DEVICE
        # ----------------------------------------------------

        batch_images = batch_images.to(
            device,
            non_blocking=True
        )

        batch_target_heatmaps = (
            batch_target_heatmaps.to(
                device,
                non_blocking=True
            )
        )

        batch_visibility = (
            batch_visibility.to(
                device,
                non_blocking=True
            )
        )


        # ----------------------------------------------------
        # CLEAR GRADIENTS
        # ----------------------------------------------------

        optimizer.zero_grad()


        # ----------------------------------------------------
        # FORWARD
        # ----------------------------------------------------

        predicted_heatmaps = my_model(
            batch_images
        )


        # ----------------------------------------------------
        # VISIBILITY MASK
        #
        # [B, 11]
        #
        # ->
        #
        # [B, 11, 1, 1]
        # ----------------------------------------------------

        visibility_mask = (
            batch_visibility
            .unsqueeze(2)
            .unsqueeze(3)
        )


        # ----------------------------------------------------
        # MSE ERROR
        # ----------------------------------------------------

        error = (

            predicted_heatmaps

            -

            batch_target_heatmaps

        ) ** 2


        # ----------------------------------------------------
        # IGNORE INVISIBLE KEYPOINTS
        # ----------------------------------------------------

        error = (
            error
            * visibility_mask
        )


        # ----------------------------------------------------
        # LOSS
        # ----------------------------------------------------

        loss = error.mean()


        # ----------------------------------------------------
        # BACKPROPAGATION
        # ----------------------------------------------------

        loss.backward()


        # ----------------------------------------------------
        # UPDATE WEIGHTS
        # ----------------------------------------------------

        optimizer.step()


        # ----------------------------------------------------
        # ACCUMULATE LOSS
        # ----------------------------------------------------

        total_loss += loss.item()


    # ========================================================
    # AVERAGE TRAIN LOSS
    # ========================================================

    average_train_loss = (

        total_loss

        /

        len(train_loader)
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    my_model.eval()

    val_total_loss = 0.0


    with torch.no_grad():

        for (
            batch_images,
            batch_target_heatmaps,
            batch_visibility
        ) in val_loader:


            # ------------------------------------------------
            # MOVE TO DEVICE
            # ------------------------------------------------

            batch_images = batch_images.to(
                device,
                non_blocking=True
            )

            batch_target_heatmaps = (
                batch_target_heatmaps.to(
                    device,
                    non_blocking=True
                )
            )

            batch_visibility = (
                batch_visibility.to(
                    device,
                    non_blocking=True
                )
            )


            # ------------------------------------------------
            # FORWARD
            # ------------------------------------------------

            predicted_heatmaps = my_model(
                batch_images
            )


            # ------------------------------------------------
            # VISIBILITY MASK
            # ------------------------------------------------

            visibility_mask = (
                batch_visibility
                .unsqueeze(2)
                .unsqueeze(3)
            )


            # ------------------------------------------------
            # MSE
            # ------------------------------------------------

            error = (

                predicted_heatmaps

                -

                batch_target_heatmaps

            ) ** 2


            # ------------------------------------------------
            # IGNORE INVISIBLE KEYPOINTS
            # ------------------------------------------------

            error = (
                error
                * visibility_mask
            )


            # ------------------------------------------------
            # LOSS
            # ------------------------------------------------

            loss = error.mean()


            val_total_loss += loss.item()


    # ========================================================
    # AVERAGE VALIDATION LOSS
    # ========================================================

    average_val_loss = (

        val_total_loss

        /

        len(val_loader)
    )


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print(

        f"Epoch: {epoch + 1:02d} | "

        f"Train Loss: "
        f"{average_train_loss:.6f} | "

        f"Val Loss: "
        f"{average_val_loss:.6f}"
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if average_val_loss < best_val_loss:

        best_val_loss = (
            average_val_loss
        )

        torch.save(

            my_model.state_dict(),

            MODEL_SAVE_PATH
        )


        print(

            f"--> Saved new best model "

            f"with Val Loss: "

            f"{best_val_loss:.6f}"
        )


# ============================================================
# TRAINING COMPLETE
# ============================================================

print()
print("=" * 60)

print("TRAINING COMPLETE")

print(
    "Best validation loss:",
    f"{best_val_loss:.6f}"
)

print(
    "Best model saved at:",
    MODEL_SAVE_PATH
)

print("=" * 60)