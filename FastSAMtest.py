from fastsam import FastSAM, FastSAMPrompt
import numpy as np
import matplotlib.pyplot as plt

image = '4035673_5'
box = [[1882, 1761, 362, 390]]
points = [[1980,1903],[2167,1683],[1727,1480]]
pointlabel = [1,1,0]
# mode = 'box'
mode = 'points'


model = FastSAM('./weights/FastSAM-x.pt')
IMAGE_PATH = f'./input/{image}.jpg'
DEVICE = 'CUDA'
everything_results = model(IMAGE_PATH, device=DEVICE, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9,)
prompt_process = FastSAMPrompt(IMAGE_PATH, everything_results, device=DEVICE)

# everything prompt
# ann = prompt_process.everything_prompt()

# text prompt
# ann = prompt_process.text_prompt(text='a photo of a dog')

if mode=='box': 
    # bbox default shape [0,0,0,0] -> [x1,y1,x2,y2]
    ann = prompt_process.box_prompt(bboxes=box)   

if mode=='points':
    # point prompt
    # points default [[0,0]] [[x1,y1],[x2,y2]]
    # point_label default [0] [1,0] 0:background, 1:foreground
    ann = prompt_process.point_prompt(points=points, pointlabel=pointlabel)

for i, mask in enumerate(ann):
    if type(mask) == dict:
        mask = mask['segmentation']

    #plt.imsave(f'./Testprojekt/pltsave_ann{i}{image}', annotation)
    plt.imsave(f'./Testprojekt/{image}_{mode}_pltsave_mask_{i}.png', mask)

    # cv2.imwrite(f'./Testprojekt/4038446-v_mask_ann{i}.jpg', annotation)
    # cv2.imwrite(f'./Testprojekt/4038446-v_mask_{i}.jpg', mask_colors)

# mask = np.zeros(ann[0], dtype=np.uint8)




prompt_process.plot(annotations=ann,output_path=f'./Testprojekt/{image}_{mode}.jpg',withContours=True)

    #annotation = mask.astype(np.uint8)

    # color = torch.rand((msak_sum, 1, 1, 3)).to(annotation.device)

    # color = np.array([0 / 255, 0 / 255, 255 / 255, 0.8])
    # annotation = annotation / 255 * color.reshape(1, 1, -1)

    # mask_colors = np.zeros((mask.shape[0],mask.shape[1], 3), dtype=np.uint8)