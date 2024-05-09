import os
import cv2
import numpy as np
import logging

from ultralytics import YOLO
from EigenCam.yolov8_cam.eigen_cam import EigenCAM
from EigenCam.yolov8_cam.utils.image import show_cam_on_image
from comet_ml import API


# Initialize logger
logging.basicConfig(level=logging.INFO)


def download_model(api_key, workspace, project, experiment_key, model_file_name):
    # Initialize Comet API
    comet_api = API(api_key=api_key)

    # Specify the experiment
    experiment = comet_api.get(workspace, project, experiment_key)

    # Get the asset id of the model file
    assets = experiment.get_asset_list()
    model_asset_id = None
    for asset in assets:
        if asset['fileName'] == model_file_name:
            model_asset_id = asset['assetId']
            break

    if model_asset_id is None:
        raise ValueError("Model file not found in the experiment assets.")

    # Get the asset data
    asset_data = experiment.get_asset(model_asset_id)

    model_file_path = f"models/{model_file_name}"
    # Write the asset data to a file
    with open(model_file_path, 'wb') as f:
        f.write(asset_data)

    return model_file_path


def process_image(file_path, model):
    # Load the image
    img = cv2.imread(file_path)

    # Resize the image
    img = cv2.resize(img, (640, 640))

    # Copy the image
    rgb_img = img.copy()

    # Normalize the image
    img = np.float32(img) / 255
    num_classes = model.model.model[-1].nc  # Get the number of classes from the model
    for i in range(1, 6):
        target_layers = [model.model.model[-i]]
        print(f"Layer {-i}:")
        print(target_layers)
    for i in range(1, 6):
        target_layers = [model.model.model[-i]]
        for class_id in range(num_classes):

            # Create EigenCAM object
            cam = EigenCAM(model, target_layers, task='seg')
            # cam = EigenGradCAM(model, target_layers )

            # Check if class_id is within the range of cam(rgb_img)'s first dimension
            if class_id < cam(rgb_img).shape[0]:
                grayscale_cam = cam(rgb_img)[class_id, :, :]  # Get the CAM for the specific class
                cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)

                # Create output directory for each class
                output_dir = os.path.join('./output/new', f"layer-{i}", f"class-{class_id}")
                os.makedirs(output_dir, exist_ok=True)

                # Save the CAM image
                output_file_path = os.path.join(output_dir,
                                                f"{os.path.basename(file_path)}_object({class_id})_heatmap_new.jpg")
                success = cv2.imwrite(output_file_path, cam_image)
                if not success:
                    logging.error(f"Failed to save image at {output_file_path}")
            else:
                break


def main():
    # model_file_path = download_model("gpludwOtJhDm2xDtgBjk5o9LT", "szittyapetro", "yolov8", "semantic_desert_6002",
    #                                "best.pt")

    # Now you can load the model
    # model = YOLO(model_file_path)
    model = YOLO("./models/best-14870.pt")

    # Specify the directory
    dir_path = './data/gtFine/images/test/bonn'

    # Get a list of all files in the directory
    file_names = os.listdir(dir_path)

    # Process each file
    for file_name in file_names:
        if ".npy" in file_name or ".json" in file_name or ".txt" in file_name:
            continue
        # Construct the full file path
        file_path = os.path.join(dir_path, file_name)

        if not os.path.exists(file_path):
            logging.warning(f"File {file_path} does not exist. Skipping.")
            continue

        logging.info(f"Processing file {file_path}...")
        process_image(file_path, model)


if __name__ == "__main__":
    main()
