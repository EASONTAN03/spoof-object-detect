# Create a script named `create_yaml.py` that generates a data.yaml for Ultralytics training
from pathlib import Path
import yaml
import shutil
from PIL import Image
import cv2
import matplotlib.image as mpimg
import os
import matplotlib.pyplot as plt

def get_config(file):
    with open(file, 'r') as read_file:
        config = yaml.safe_load(read_file)
    raw_data_path = Path(config['data']['raw'])
    processed_data_path = Path(config['data']['processed'])

    models_path = Path(config['outputs']['models'])
    predict_path = Path(config['outputs']['predict'])   

    datasets_path = raw_data_path / config['configs']['dataset']
    model_name = config['configs']['model_name']
    data_class = config['configs']['class']
    prepare_benchmark = str(config['configs']['prepare_benchmark'])
    model_benchmark = str(config['configs']['model_benchmark'])

    params = config['params']

    return datasets_path, processed_data_path, models_path, predict_path, model_name, data_class, prepare_benchmark, model_benchmark, params

def count_and_average_image_size(split_path):
    image_files = list(split_path.rglob("*.*"))
    count = len(image_files)

    total_width = 0
    total_height = 0
    min_width = float('inf')
    max_width = 0
    min_height = float('inf')
    max_height = 0

    for img_file in image_files:
        with Image.open(img_file) as img:
            width, height = img.size
            total_width += width
            total_height += height

            min_width = min(min_width, width)
            max_width = max(max_width, width)
            min_height = min(min_height, height)
            max_height = max(max_height, height)

    avg_width = total_width / count if count else 0
    avg_height = total_height / count if count else 0

    if count == 0:
        min_width = min_height = max_width = max_height = 0

    return count, (avg_width, avg_height), (min_width, min_height, max_width, max_height)

def resize_and_adjust_labels(target_size, img_path, label_path, out_img_path, out_label_path):
    with Image.open(img_path) as img:
        w_orig, h_orig = img.size
        img_resized = img.resize(target_size)
        img_resized.save(out_img_path)

    scale_x = target_size[0] / w_orig
    scale_y = target_size[1] / h_orig

    if label_path.exists():
        with open(label_path, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            cls, xc, yc, bw, bh = map(float, line.strip().split())
            xc = xc * scale_x
            yc = yc * scale_y
            bw = bw * scale_x
            bh = bh * scale_y
            xc = min(max(xc / target_size[0], 0), 1)
            yc = min(max(yc / target_size[1], 0), 1)
            bw = min(max(bw / target_size[0], 0), 1)
            bh = min(max(bh / target_size[1], 0), 1)

            # Save with 6 decimal places
            new_lines.append(f"{int(cls)} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        with open(out_label_path, 'w') as f:
            f.write("\n".join(new_lines))

def draw_bbox(image_path, label_path, show=True):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Image not found: {image_path}")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    label_path = Path(label_path)
    if not label_path.exists():
        print(f"Label file not found: {label_path}")
        return

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        try:
            cls, xc, yc, bw, bh = map(float, line.strip().split())
        except ValueError:
            cls, xc, yc, bw, bh, conf = map(float, line.strip().split())
        # Convert normalized YOLO format to pixel coordinates
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)

        color = (0, 255, 0) if int(cls) == 0 else (255, 0, 0)
        label = '0: Live' if int(cls) == 0 else '1: Spoof'
        cv2.rectangle(image_rgb, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image_rgb, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Plot tightly with no borders or padding
    if show==True:
        fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])  # Full image
        ax.imshow(image_rgb)
        ax.axis("off")
        plt.show()
    else:
        return image_rgb

def show_grid_images(image_paths, label_dir, n_rows=5, n_cols=4):
    plt.figure(figsize=(15, 10))
    for idx, img_path in enumerate(image_paths):
        label_filename = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        label_path = os.path.join(label_dir, label_filename)

        img_with_bbox = draw_bbox(img_path, label_path, show=False)

        plt.subplot(n_rows, n_cols, idx + 1)
        if img_with_bbox is not None:
            plt.imshow(img_with_bbox)
        else:
            plt.imshow(mpimg.imread(img_path))
        plt.axis('off')
        plt.title(os.path.basename(img_path), fontsize=8)

    plt.tight_layout()
    plt.show()

def create_data_yaml(train_path,val_path,test_path,class_names=["live", "spoof"],save_path="data.yaml"):
    data = {
        "train": train_path,
        "val": val_path,
        "test": test_path,
        "nc": len(class_names),
        "names": class_names
    }

    # Save to YAML
    with open(save_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

    print(f"dataset.yaml created at: {Path(save_path).resolve()}")

def trained_results_split(
    model_path,  # or path to your YOLO training output
    plots_dir,
    images_dir
):
    # Create output directories if they don't exist
    
    plots_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    # Move plot .png files
    for file in model_path.glob("*.png"):
        shutil.move(str(file), plots_dir / file.name)

    # Move visual sample images (.jpg)
    for file in model_path.glob("*.jpg"):
        shutil.move(str(file), images_dir / file.name)

    print(" Training results split and saved to:")
    print(f"   Models: {model_path.resolve()}")
    print(f"   Plots:  {plots_dir.resolve()}")
    print(f"   Images: {images_dir.resolve()}")


# Only run when executed directly (not when imported)
if __name__ == "__main__":
    create_data_yaml(train_path="train/images",
                    val_path="val/images",
                    test_path="test/images",
                    class_names=["live", "spoof"],
                    save_path="data.yaml")
    
    draw_bbox(image_path="test/images/sample.jpg",
              label_path="test/labels/sample.txt")  
