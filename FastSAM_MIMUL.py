import os
from fastsam import FastSAM, FastSAMPrompt
import numpy as np
import matplotlib.pyplot as plt
import argparse
from utils.tools import convert_box_xywh_to_xyxy
import ast

def get_arguments():

    parser = argparse.ArgumentParser(prog='MIMUL FastSAM', description='Implements FastSAM for use on MIMUL Piano Roll heads.')
    parser.add_argument('-w', '--working_directory', type=str, required=True, help='Path to the working directory with inputs and outpus.')
    parser.add_argument('-ma', '--manufacturer', type=str, required=False, help='The piano roll manufacturer. This argument is for better sorting while testing')
    parser.add_argument('-i', '--input_image', type=str, required=True, help='The file name of the picture (without extention) to be segmented. (Only tested with JPG.)')
    parser.add_argument('-t', '--target', type=str, required=True, help='The target that should be segmented. This is for sorting during the test phase.')
    parser.add_argument('-m', '--mode', type=str, required=True, help='The mode to use for manually marking the location of the label or licence stamp. For box mode type \'box\'. for points mode use \'points\'.')
    parser.add_argument('-b', '--box', type=str, required=False, help='The box of the label or stamp. Requires format like in FastSAM: bbox default shape [0,0,0,0] -> [x1,y1,x2,y2] Example: box = [[1692, 882, 440, 508]]')
    parser.add_argument('-p', '--points', type=str, required=False, help='The points of the label or stamp. Requires format like in FastSAM: points default [[0,0]] [[x1,y1],[x2,y2]]')
    parser.add_argument('-pl', '--point_labels', type=str, required=False, help='The point_label to define which points belong to the foreground and which belong to the background. Requires format like in FastSAM: point_label default [0] [1,0] 0:background, 1:foreground')
    args = parser.parse_args()
    print(f'args={args}')

    print(f'Image to be processed: {args.input_image}.jpg')
    if(args.mode == 'box' and args.box != None):
        print(f'Box mode chosen; box={args.box}')
        args.box = convert_box_xywh_to_xyxy(ast.literal_eval(args.box))
    elif(args.mode == 'box' and args.box == None):
        parser.error(f'Box mode requires the --box argument')
    elif(args.mode == 'points' and args.points !=None and args.point_labels != None):
        args.points = ast.literal_eval(args.points)
        args.point_labels = ast.literal_eval(args.point_labels)
        print(f'points mode chosen; points={args.points}, point_labels={args.point_labels}')
    elif(args.mode == 'points' and args.points == None):
        parser.error('Points mode requires the --points argument')
    elif(args.mode == 'points' and args.point_labels == None):
        parser.error('Points mode requires the --point_labels argument')
    return args

def main():

    args = get_arguments()

    model = FastSAM('./weights/FastSAM-x.pt')
    image_path = f'{args.working_directory}/{args.manufacturer}/Input/{args.input_image}.jpg'
    output_path = f'{args.working_directory}/{args.manufacturer}/Output/{args.target}/{args.mode}'
    DEVICE = 'CUDA'
    everything_results = model(image_path, device=DEVICE, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9,)
    prompt_process = FastSAMPrompt(image_path, everything_results, device=DEVICE)

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
        ann = prompt_process.point_prompt(points=list(args.points), pointlabel=list(args.point_labels))

    #save mask
    for i, mask in enumerate(ann):
        if type(mask) == dict:
            mask = mask['segmentation']
        os.makedirs(f'{output_path}/Masks/', exist_ok=True)
        print(f"saving annotations mask {i} to folder {output_path}/Masks/")
        plt.imsave(f'{output_path}/Masks/{args.input_image}.png', mask)

    #save reference image
    os.makedirs(f'{output_path}/Images/', exist_ok=True)
    print(f"Promting FastSAM for ")
    prompt_process.plot(annotations=ann, output_path=f'{output_path}/Images/{args.input_image}.jpg',withContours=True)

if __name__ == "__main__":
    main()