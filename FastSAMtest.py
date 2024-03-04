from fastsam import FastSAM, FastSAMPrompt
import numpy as np
import matplotlib.pyplot as plt
import argparse
from utils.tools import convert_box_xywh_to_xyxy
import ast

def get_arguments():

    parser = argparse.ArgumentParser(prog='MIMUL FastSAM', description='Implements FastSAM for use on MIMUL Piano Roll heads.')
    parser.add_argument('-i', '--image_input', type=str, required=True, help='The name (without extension) of the picture in question. So far this only works for jpg.')
    # parser.add_argument('-o', '--output_folder', type=str, required=True, help='The output folder for the annotated picture and its annotation mask to be safed in.')
    parser.add_argument('-m', '--mode', type=str, required=True, help='The mode to use for manually marking the location of the label or licence stamp. For box mode type \'box\'. for points mode use \'points\'.')
    parser.add_argument('-b', '--box', type=str, required=False, help='The box of the label or stamp. Requires format like in FastSAM: bbox default shape [0,0,0,0] -> [x1,y1,x2,y2] Example: box = [[1692, 882, 440, 508]]')
    parser.add_argument('-p', '--points', type=str, required=False, help='The points of the label or stamp. Requires format like in FastSAM: points default [[0,0]] [[x1,y1],[x2,y2]]')
    parser.add_argument('-pl', '--point_labels', type=str, required=False, help='The point_label to define which points belong to the foreground and which belong to the background. Requires format like in FastSAM: point_label default [0] [1,0] 0:background, 1:foreground')
    args = parser.parse_args()
    print(f'args={args}')

    print(f'Image to be processed: {args.image_input}.jpg')
    if(args.mode == 'box' and args.box != None):
        print(f'Box mode chosen; box={args.box}')
        args.box = convert_box_xywh_to_xyxy(ast.literal_eval(args.box))
    elif(args.mode == 'box' and args.box == None):
        parser.error(f'Box mode requires the --box argument')
    elif(args.mode == 'points' and args.points !=None and args.point_label != None):
        print(f'points mode chosen; points={args.points}, point_label={args.point_label}')
    elif(args.mode == 'points' and args.points == None):
        parser.error('Points mode requires the --points argument')
    elif(args.mode == 'points' and args.point_label == None):
        parser.error('Points mode requires the --point_labels argument')
    return args


# image = '4035713_32'
# box = [[1692, 882, 440, 508]]
# points = [[1980,1903],[2167,1683],[1727,1480]]
# pointlabel = [1,1,0]
# mode = 'box'
# mode = 'points'

def main():

    args = get_arguments()

    model = FastSAM('./weights/FastSAM-x.pt')
    IMAGE_PATH = f'./input/{args.image_input}.jpg'
    DEVICE = 'CUDA'
    everything_results = model(IMAGE_PATH, device=DEVICE, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9,)
    prompt_process = FastSAMPrompt(IMAGE_PATH, everything_results, device=DEVICE)

    # everything prompt
    # ann = prompt_process.everything_prompt()

    # text prompt
    # ann = prompt_process.text_prompt(text='a photo of a dog')

    if args.mode=='box': 
        # bbox default shape [0,0,0,0] -> [x1,y1,x2,y2]
        ann = prompt_process.box_prompt(bbox=list(args.box))   

    if args.mode=='points':
        # point prompt
        # points default [[0,0]] [[x1,y1],[x2,y2]]
        # point_label default [0] [1,0] 0:background, 1:foreground
        ann = prompt_process.point_prompt(points=args.points, pointlabel=args.point_labels)

    for i, mask in enumerate(ann):
        if type(mask) == dict:
            mask = mask['segmentation']

        plt.imsave(f'./output/{args.image_input}_{args.mode}_pltsave_mask_{i}.png', mask)

    prompt_process.plot(annotations=ann,output_path=f'./output/{args.image_input}_{args.mode}.jpg',withContours=True)

if __name__ == "__main__":
    main()