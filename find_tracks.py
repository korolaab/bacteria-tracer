#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os
#os.environ["CUDA_VISIBLE_DEVICES"]=""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tracker_imporved import Tracker
import sys
#from model import Unet

# model = Unet(input_shape=(300,300,1),pretrained_weights="model_2_Deeper.h5")
# Open Video

#name = "/home/korolaab/Works/bacteria_tracer/videos_coli/jM109OD2-4_first290frames.avi"
name = sys.argv[0]
cap    = cv2.VideoCapture(name)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # original file width
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # original file height
fps    = int(cap.get(cv2.CAP_PROP_FPS))
codec  = cv2.VideoWriter_fourcc(*'XVID') # video codec for processed video with points
print("%4sx%4s %4s"%(width,height,fps))

#model = Unet(input_shape=(height,width,1),pretrained_weights="model_2_Deeper.h5")          
#model.summary()
## Randomly select 25 frames
frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) 
print(frameIds)
# Store selected frames in an array
frames = []
for fid in range(0,int(frameIds),10):
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = cap.read()
    frames.append(frame)

# Calculate the median along the time axis
medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)
del frames
cap.release()


# In[5]:



cap    = cv2.VideoCapture(name)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("track_DL.avi", fourcc, fps, (width,height),True) # outpur video object

out2 = cv2.VideoWriter("track_mask_DL.avi", fourcc, fps, (width,height),True) # outpur video object
      
        
def finderCV(mask,criteria=0.8,img=None):
            mask = np.where(mask > criteria, 1, 0)
            centroids = []
            params = []
            im1 = np.ascontiguousarray(mask, dtype=np.uint8)
            contours, hierarchy = cv2.findContours(im1, cv2.RETR_TREE , cv2.CHAIN_APPROX_NONE)
            for cnt in contours:
                    #(x,y),r = cv2.minEnclosingCircle(cnt)
                    (x,y),(w,h),angle=cv2.minAreaRect(cnt)
                    r = (w**2+h**2)/2
                    
                    
                    if(w<h):
                        angle= angle-90
                    else:
                        w,h = h,w
                    

                    
                    if(r>=5):
                        b = np.array([[x],[y]])
                        centroids.append(b)
                        params.append([angle,w,h])
                        
            return centroids,params


grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)

# Loop over all frames
ret = True

l = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(l)

# Create Object Tracker
tracker = Tracker(100, 30, 30, 0)
track_colors = [(255, 0, 0), (0, 255, 0), (255, 255, 0),
                    (0, 255, 255), (255, 0, 255), (255, 127, 255),
                    (127, 0, 255), (127, 0, 127)]               
folder ="tracks"
ret,frame=cap.read()
for frame_number in range(0,l):

      # Read frame
    ret, frame = cap.read()
      # Convert current frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      # Calculate absolute difference of current frame and 
      # the median frame
    dframe = cv2.absdiff(gray, grayMedianFrame)

    
    # Treshold to binarize
    th, dframe = cv2.threshold(dframe, 30, 255, cv2.THRESH_BINARY)
    dframe = cv2.blur(dframe,(2,2))

    centers,params = finderCV(dframe)
    #  If centroids are detected then track them
    #print(len(centers))
    f=open(folder+"/%03d.csv"%frame_number,'w')
    print("id,x,y,angle,width,length,is_lost",end='\n',file=f)

    d = cv2.cvtColor((dframe).astype(np.uint8),  cv2.COLOR_GRAY2BGR)
    
    if (len(centers)>0):
        
         #  Track object using Kalman Filter
        tracker.Update(centers,params)

        #           For identified object tracks draw tracking line
        #            Use various colors to indicate different track_id
        
        
        
        for i in range(len(tracker.tracks)):
            
            if (len(tracker.tracks[i].trace) > 1 ):
                for j in range(len(tracker.tracks[i].trace)-1):
                                # Draw trace line
                    x1 = tracker.tracks[i].trace[j][0][0]
                    y1 = tracker.tracks[i].trace[j][1][0]
                    x2 = tracker.tracks[i].trace[j+1][0][0]
                    y2 = tracker.tracks[i].trace[j+1][1][0]
                    
                    
                    clr = tracker.tracks[i].track_id % 8
                    
                    if(np.sqrt((x1-x2)**2 + (y1-y2)**2)<=8 and tracker.tracks[i].is_lost == 0 ):
                        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                             track_colors[clr], 1)
                        cv2.line(d, (int(x1), int(y1)), (int(x2), int(y2)),
                                             track_colors[clr], 1)
                if(tracker.tracks[i].is_lost == 0 ):  
                    cv2.putText(frame, "%d"%tracker.tracks[i].track_id, (int(x1),int(y1)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, track_colors[clr],1)
                    
                    cv2.putText(d, "%d"%tracker.tracks[i].track_id, (int(x1),int(y1)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, track_colors[clr],1)
                
                angle = tracker.tracks[i].angle
                width = tracker.tracks[i].width
                length = tracker.tracks[i].length
                is_lost = tracker.tracks[i].is_lost
                
                if(np.sqrt((x1-x2)**2 + (y1-y2)**2)<=8):
                    print("%d,%d,%d,%f,%f,%f,%d"%(tracker.tracks[i].track_id,x2,y2,angle,
                                                  width,length,is_lost),
                          end='\n',file=f)
                    

        f.close()

    out.write(frame)


      
    out2.write(d) 
# Release video object
cap.release()
out.release()
out2.release()

