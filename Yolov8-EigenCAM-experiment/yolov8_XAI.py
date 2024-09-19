# Description: This script is used to generate explanations for the YOLOv8 model using EigenCAM and LIME.
from pathlib import Path
import cv2
import numpy as np
import logging
from ultralytics import YOLO
from EigenCam.yolov8_cam.eigen_cam import EigenCAM
from EigenCam.yolov8_cam.utils.image import show_cam_on_image # noqa
from LIME import YoloLimeExplainer
from comet_ml import API

# Initialize logger
logging.basicConfig(level=logging.INFO)


def download_model(api_key, workspace, project, experiment_key, model_file_name):
    """
    Download the model file from Comet.ml
    :param api_key:
    :param workspace:
    :param project:
    :param experiment_key:
    :param model_file_name:
    :return:
    """
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


def process_eigen_cam(file_path, model):
    """
    Process the image and generate CAMs for each class in the image using EigenCAM and EigenGradCAM.
    :param file_path:
    :param model:
    :return:
    """
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
        for class_id in range(num_classes):

            # Create EigenCAM object
            cam = EigenCAM(model, target_layers, task='seg')

            # Check if class_id is within the range of cam(rgb_img)'s first dimension
            if class_id < cam(rgb_img).shape[0]:
                grayscale_cam = cam(rgb_img)[class_id, :, :]  # Get the CAM for the specific class
                cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)

                # Create output directory for each class
                output_dir = file_path.parent / "explain_output"/"eigencam"
                output_dir.mkdir(parents=True, exist_ok=True)
                output = output_dir / f"{file_path.name}layer-{i}" / f"class-{class_id}"
                output.mkdir(parents=True, exist_ok=True)

                # Save the CAM image
                output_file_path = output / f"{file_path.name}_object({class_id})_heatmap.jpg"
                success = cv2.imwrite(output_file_path, cam_image)
                if not success:
                    logging.error(f"Failed to save image at {output_file_path}")
            else:
                break


def process_image(file_path: Path, model):
    """
    Process the image and generate explanations using both EigenCAM and LIME. The explanations are saved in the explain_output
    directory.
    :param file_path: Path to the image file.
    :param model: YOLO model.
    """
    # Process EigenCAM
    process_eigen_cam(file_path, model)

    # Initialize LIME explainer
    lime_explain = YoloLimeExplainer(model=model)

    # Process LIME
    lime_explain(image=file_path, predict_fn=lime_explain.yolo_segmentation_predict)


def main(arguments):
    """
    Main function of YOLOv8 XAI
    :return:
    """
    # Specify whether to use the local model or download from Comet.ml
    if not arguments.use_local_model:
        model_file_path = download_model("gpludwOtJhDm2xDtgBjk5o9LT", "szittyapetro", "yolov8", "semantic_desert_6002",
                                         "best.pt")
        model = YOLO(model_file_path)
    else:
        model = YOLO("models/best-14870.pt")
    # Specify the output directory
    output_dir = Path("./output/explain")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Specify the directory
    dir_path = Path('./data/gtFine/images/test/bonn')

    # Get a list of all files in the directory
    file_names = dir_path.iterdir()

    # Process each picture in the directory
    for file_name in file_names:
        if not file_name.is_file() or file_name.suffix not in [".png", ".jpg"]:
            continue
        # Construct the full file path
        file_path = file_name

        if not file_path.exists():
            logging.warning(f"File {file_path} does not exist. Skipping.")
            continue

        logging.info(f"Processing file {file_path}...")
        process_image(file_path, model)


def parse_args():
    """
    Parse command line arguments
    :return:
    """
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv8 XAI")
    parser.add_argument("--use-local-model", action="store_true",
                        help="Use the local model instead of downloading from Comet.ml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
