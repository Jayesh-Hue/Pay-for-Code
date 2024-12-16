#Import libraties
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw

# Function to calculate IoU 
def calculate_iou(box1, box2):
    # Calculate intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate intersection area
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    # Calculate areas of both bounding boxes
    area_box1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area_box2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # Calculate union area
    union_area = area_box1 + area_box2 - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area
    return iou if union_area > 0 else 0

# Process each file
file_names = ['006037', '006042', '006048', '006054', '006059', '006067', '006097', '006098','006206', '006211' ,'006227', '006253', '006291', '006310', '006312', '006315', '006329', '006374']
for i in file_names:
    # Load the ground truth bounding boxes
    gt_boxes = np.loadtxt(f"./KITTI_Selection/labels/{i}.txt", usecols=(1, 2, 3, 4)).tolist()
    
    # Path to the image
    img_path = f"./KITTI_Selection/images/{i}.png"

    # Create YOLO model (consider pre-trained models with higher accuracy)
    model = YOLO("yolov8x.pt") 
    
     # Object detection with adjustments for accuracy
    results = model.predict(
        source=img_path,
        save=True,  # Optional: Save image with bounding boxes
        conf=0.25,  # Increase confidence threshold for stricter detection (0.3 to 0.5)
        augment=True,)
    
    # Extract detection results
    boxes = results[0].boxes.xyxy.tolist()
    confidences = results[0].boxes.conf.tolist()
    
#     # Iterate over all result images in 'results' if processing multiple images
# for result in results:
#     boxes = result.boxes.xyxy.tolist()  # Bounding boxes for the current image
#     confidences = result.boxes.conf.tolist()  # Confidence scores for the current image

# Filter out detections based on IoU with ground truth boxes
    filtered_boxes = []
    for box, conf in zip(boxes, confidences):
        x1, y1, x2, y2 = box
        detected_box = [x1, y1, x2, y2]
    for gt_box in gt_boxes:
        print("detected_box:", detected_box, "gt_box:", gt_box) 
        # Calculate IoU with each ground truth box
        max_iou = max([calculate_iou(detected_box, gt_box) for gt_box in gt_boxes])
        if max_iou >= 0.45:
            filtered_boxes.append(box)
    
    
    # Save filtered bounding box coordinates
    output_file = f"./KITTI_Selection/bb_coordinates/{i}.txt"
    with open(output_file, 'w') as f:
        for box in filtered_boxes:
            x1, y1, x2, y2 = box
            f.write(f"{x1} {y1} {x2} {y2}\n")
            
    # Display the image with YOLO detection and ground truth boxes overlaid
    image_combined = Image.fromarray(results[0].plot()[:,:,::-1])
    draw = ImageDraw.Draw(image_combined)
    for gt_box in gt_boxes:
        x1, y1, x2, y2 = gt_box
        draw.rectangle([x1, y1, x2, y2], outline="green", width=3)

    # Display the image with filtered YOLO detection bounding boxes only
    image_filtered = Image.open(img_path)
    draw = ImageDraw.Draw(image_filtered)
    for box in filtered_boxes:
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    #image_filtered.show()