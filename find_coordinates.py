#! /usr/bin/python3
import warnings
warnings.filterwarnings("ignore")
import os # working with os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # no debug TF info
from keras.models import load_model # lib for easier NN coding with Tensorflow
import tensorflow as tf # high level framework for NN
import numpy as np #matrixes and tensors 
import cv2 
import json # JSON lib
import math # math module 
from progress_bar import printProgressBar # beautiful progress bar :3
import model 
import argparse # parsing input argument

def recognize(image): #get mask of bacteria on one frame of video
    shift = 100 # step for sliding window
    size = 100 # sliding window size
    image = np.expand_dims(image,axis=2)/255 # adding 1 diminision to array
    output = np.zeros([image.shape[0],image.shape[1]],dtype="float32") # raw output_mask
    buf1 = np.zeros([1,size,size,1],dtype="float32") # buffer 1
    buf2 = np.zeros([size,size],dtype="float32") # buffer 2
    buf3=np.zeros([size,size],dtype="float32") # buffer 3
    for x_shift in range(math.ceil(image.shape[0]/shift)): # processing frame with sliding window
        for y_shift in range(math.ceil(image.shape[1]/shift)):
            x = shift*x_shift
            y = shift*y_shift
            piece = image[x:x+size,y:y+size,:] # piece of image
            buf1[0,:piece.shape[0],:piece.shape[1],:]=piece
            buf2 = (model.predict(buf1))[0,:,:,0] # predicting for piece of image
            output[x:x+piece.shape[0],y:y+piece.shape[1]]+=buf2[:piece.shape[0],:piece.shape[1]]
    return output



def get_centre_of_shapes(mask): # get centre of all finded bacteria on frame
    
    mask = np.where(mask>0.35,1,0) # Now we decide, that all pixel with value 0.3 is bacteria. In future we should fix mistakes of NN-inferece/
  
    #mask = np.ascontiguousarray(mask, dtype=np.uint8) # float32 -> int8 
    contours,_ = cv2.findContours(mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE) # finding bacteria's contours (pixels where each bacteria lays)
    
    coordinates=[] # list for points where bacterias lay
    
    for cnt in contours:
        cX = 0 # sum X
        cY = 0 # sum Y
        l = len(cnt) # total number of pixels where bacteria lays
        for kp in cnt: # point where bacteria lays is average of x,y coordinates of bacteria's contour
            cX = cX+kp[0][0] # kp[0][0] - X coordinate
            cY = cY+kp[0][1] # kp[0][1] - Y coordinate
        coordinates.append([int(cX/l),int(cY/l)]) # [avg(X), avg(Y)]
    return coordinates 


def video_processing(filename, output_video):
    cap = cv2.VideoCapture(filename) #input video file object
    
    if(output_video):
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # original file width
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # original file height
        fps = int(cap.get(cv2.CAP_PROP_FPS)) # original file FPS
        fourcc = cv2.VideoWriter_fourcc(*'DIVX') # video codec for processed video with points
        out = cv2.VideoWriter(output_video,fourcc,fps,(width,height)) # outpur video object
        
    
    coordinate_from_all_frames={} #dictionary where for each frame we will save coordinate of bacteria
    
    l = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # total number of frames in video
    print("Frames in video: %d"%(l))
    printProgressBar(0,l,prefix="Progress:",suffix="Complete",length=50) # begin of beautiful progress bar :3
    for frame_number in range(0,l):
        retval,frame = cap.read() # reading one frame
        gray_frame = frame[:,:,0] # converting RGB -> Grayspace
        segmentation_map = recognize(gray_frame) # inference prediction for frame
        coordinates_from_one_frame=get_centre_of_shapes(segmentation_map) # coordinates of bacteria for frame
        if(output_video): # if you want to save video with marked bacterias 
            for point in coordinates_from_one_frame:                
                cv2.circle(frame, (point[0],point[1]), 3, (255, 0, 0), -1) # draw blue point on frame
            frame[:,:,1]=frame[:,:,1]+segmentation_map*200 # green 
            out.write(frame)
        coordinate_from_all_frames[str(frame_number)]=coordinates_from_one_frame
        printProgressBar(frame_number+1,l,prefix="Progress:",suffix="Complete",length=50) # update of beautiful progress bar :3
    cap.release()
    if(out_video):
        out.release()
    return coordinate_from_all_frames


if __name__ == "__main__": #if you execute this file
    parser = argparse.ArgumentParser(description="Find bacterias coordingates on video.")
    parser.add_argument("--input",action="store",metavar="<path>",required=True,dest="filename", help="input video file")
    parser.add_argument("--output",action="store",metavar="<path>",
   required=True,dest="output_coord_filename",help="output json file")
    parser.add_argument("--output_video",action="store",dest="output_video_filename", default=None,metavar="<path>",help="ouptut video with marked bacterias")
    
    args=parser.parse_args()
    
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())
    model = load_model('model.h5',custom_objects={'focal_loss': 'mse'})

    coord = video_processing(args.filename,args.output_video_filename)
    with open(args.output_cord_filename,"w") as f:
        f.write(json.dumps({"video_filename":args.filename,"frames":coord}))


