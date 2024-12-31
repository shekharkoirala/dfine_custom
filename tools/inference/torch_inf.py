"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torchvision.transforms as T

import numpy as np
from PIL import Image, ImageDraw

import sys
import os
import cv2  # Added for video processing

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.core import YAMLConfig


def draw(images, labels, boxes, scores, thrh=0.4):
    class_map = {0: 'person', 1: 'Car', 2: 'Truck', 3: 'Bus', 4: 'Van', 5: 'Military_Vehicle', 6: 'Pickup_Truck', 7: 'Motorbike', 8: 'Bicycle', 9: 'Boat', 10: 'Airplane', 11: 'Helicopter', 12: 'Ship', 13: 'unknown', 14: 'no_idea'}
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)

        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scr[scr > thrh]

        for j, b in enumerate(box):
            draw.rectangle(list(b), outline='red')
            draw.text(
                (b[0], b[1]),
                text=f"{class_map[lab[j].item()]} {round(scrs[j].item(), 2)}",
                fill='blue',
            )

        im.save('torch_results.jpg')


def process_image(model, device, file_path):
    im_pil = Image.open(file_path).convert('RGB')
    w, h = im_pil.size
    orig_size = torch.tensor([[w, h]]).to(device)

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    im_data = transforms(im_pil).unsqueeze(0).to(device)

    output = model(im_data, orig_size)
    labels, boxes, scores = output

    draw([im_pil], labels, boxes, scores)


def process_video(model, device, file_path):
    cap = cv2.VideoCapture(file_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('torch_results.mp4', fourcc, fps, (orig_w, orig_h))

    frame_count = 0
    print("Processing video frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (640, 640))
        im_data = torch.from_numpy(frame_resized).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
        orig_size = torch.tensor([[orig_w, orig_h]]).to(device)

        with torch.no_grad():
            output = model(im_data, orig_size)
        labels, boxes, scores = output

        # Convert frame to PIL for drawing
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Draw detections
        draw([frame_pil], [labels], [boxes], [scores], thrh=0.4)

        # Convert back to OpenCV image
        frame_with_detections = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

        # Write the frame to the output video
        out.write(frame_with_detections)
        frame_count += 1

        if frame_count % 100 == 0:
            torch.cuda.empty_cache()
            print(f"Processed {frame_count} frames...")

        # if frame_count == 3000:
        #     break

    cap.release()
    out.release()
    print("Video processing complete. Result saved as 'results_video.mp4'.")


def main(args):
    """Main function"""
    cfg = YAMLConfig(args.config, resume=args.resume)

    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')

    # Load train mode state and convert to deploy mode
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    device = args.device
    model = Model().to(device)

    # Check if the input file is an image or a video
    file_path = args.input
    if os.path.splitext(file_path)[-1].lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
        # Process as image
        process_image(model, device, file_path)
        print("Image processing complete.")
    else:
        # Process as video
        process_video(model, device, file_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-r', '--resume', type=str, required=True)
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-d', '--device', type=str, default='cpu')
    args = parser.parse_args()
    main(args)