from fastsam import FastSAM, FastSAMPrompt
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os

def maskpreview(annotations):
    for i, mask in enumerate(annotations):
        annotation = mask.astype(np.uint8)
        plt.imshow(annotation)
        plt.show()

def maskprinttofile(annotations, output_path):
    # result
    for i, mask in enumerate(annotations):
        annotation = mask.astype(np.uint8)
        plt.imshow(annotation)
        plt.axis('off')
        plt.show()
        fig = plt.gcf()
        plt.draw()

        try:
            buf = fig.canvas.tostring_rgb()
        except AttributeError:
            fig.canvas.draw()
            buf = fig.canvas.tostring_rgb()
        cols, rows = fig.canvas.get_width_height()
        img_array = np.frombuffer(buf, dtype=np.uint8).reshape(rows, cols, 3)
        result = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        plt.close()
    
    path = os.path.dirname(os.path.abspath(output_path))
    if not os.path.exists(path):
        os.makedirs(path)
    # result = result[:, :, ::-1]
    cv2.imwrite(output_path, result)    
    


model = FastSAM('./weights/FastSAM-x.pt')
IMAGE_PATH = './images/4038446-v.jpg'
output_path = './Testprojekt/4038446-v.jpg'
DEVICE = 'cpu'
# DEVICE = 'CUDA'
everything_results = model(IMAGE_PATH, device=DEVICE, retina_masks=True, imgsz=3360, conf=0.4, iou=0.9,)
prompt_process = FastSAMPrompt(IMAGE_PATH, everything_results, device=DEVICE)

# everything prompt
# ann = prompt_process.everything_prompt()

# bbox default shape [0,0,0,0] -> [x1,y1,x2,y2]
#ann = prompt_process.box_prompt(bboxes=[[600,1590,335,324],[33,1430,556,611]])
ann = prompt_process.box_prompt(bboxes=[[33,1430,556,611],[600,1590,335,324]])
print("Size: ", ann.size)
print("Shape: ", ann.shape)
print("Dimension: ", ann.ndim)
print("Datatype: ", ann.dtype)

maskpreview(ann)
maskprinttofile(ann, output_path)

# text prompt
# ann = prompt_process.text_prompt(text='a photo of a dog')

# point prompt
# points default [[0,0]] [[x1,y1],[x2,y2]]
# point_label default [0] [1,0] 0:background, 1:foreground
# ann = prompt_process.point_prompt(points=[[620, 360]], pointlabel=[1])

prompt_process.plot(annotations=ann,output_path='./output/4038446-v.jpg',)

# python Inference.py --model_path ./weights/FastSAM-x.pt --img_path ./images/4038543.jpg  --box_prompt "[[457,1078,297,297]]"


