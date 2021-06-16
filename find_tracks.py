#!/usr/bin/python3
import os
#os.environ["CUDA_VISIBLE_DEVICES"]=""
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tracker_imporved import Tracker
import sys
from progress_bar import printProgressBar
from argparse import ArgumentParser
import datetime
#name = sys.argv[1]
#output_name = sys.argv[2]
#format_output =0 

name = ''
output_name = ''
format_output = 0
#### Parameters
THRESH = 30
DIST_THRESH = 100
MAX_FRAMES_TO_SKIP = 30
MAX_TRACE_LENGTH = 30
TRACK_ID_COUNT = 0

####
if __name__ == "__main__":
    parser = ArgumentParser("tracks finder")
    parser.add_argument("-i",dest='name',action='store', required=True, help='Input video file')
    parser.add_argument("-o",dest='output_name',action='store', required=True, help='output_name')
    parser.add_argument("--thresh",dest='thresh',action='store', help='pixels level threshold')
    parser.add_argument("--dist_thresh",dest='dist_thresh',action='store',default=100, help = 'distance threshold. When exceeds the threshold, track will be deleted and new track is created')
    parser.add_argument("--max_frames_to_skip",dest='max_frames_to_skip',action='store', default = 30, help ='maximum allowed frames to be skipped for the track object undetected')
    parser.add_argument("--max_trace_length",dest='max_trace_length',action='store',default = 30,  help='trace path history length')
    parser.add_argument("--trackIdCount",dest='trackIdCount',action='store', default = 0, help='identification of each track object')
    args = parser.parse_args()
    
    name = args.name
    output_name = args.output_name
    THRESH = float(args.thresh)
    DIST_THRESH = args.dist_thresh
    MAX_TRACE_LENGTH = args.max_trace_length
    MAX_FRAMES_TO_SKIP = args.max_frames_to_skip
    TRACK_ID_COUNT = args.trackIdCount




str_datetime = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
TMP_tracks = 'tracks_{}'.format(str_datetime)

if len(name) < 1:
    print("no filename")
    exit()
if len(output_name) < 1:
    print("no output filename")
    exit()
    
try:
    os.mkdir(TMP_tracks)
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise
    pass

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
out = cv2.VideoWriter("/tmp/tracks_{}.avi".format(str_datetime), fourcc, fps, (width,height),True) # outpur video object

out2 = cv2.VideoWriter("/tmp/tracks_mask_{}.avi".format(str_datetime), fourcc, fps, (width,height),True) # outpur video object
      
        
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
#print(l)

# Create Object Tracker


tracker = Tracker(DIST_THRESH,
                  MAX_FRAMES_TO_SKIP,
                  MAX_TRACE_LENGTH,
                  TRACK_ID_COUNT)
track_colors = [(255, 0, 0), (0, 255, 0), (255, 255, 0),
                    (0, 255, 255), (255, 0, 255), (255, 127, 255),
                    (127, 0, 255), (127, 0, 127)]               
folder =TMP_tracks

for frame_number in range(0,l):
      # Read frame
    printProgressBar(frame_number,l,folder,length = 30) 
    ret, frame = cap.read() 
    # Convert current frame to grayscale 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      # Calculate absolute difference of current frame and 
      # the median frame
    dframe = cv2.absdiff(gray, grayMedianFrame)
    # Treshold to binarize
    th, dframe = cv2.threshold(dframe, THRESH, 255, cv2.THRESH_BINARY)
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
printProgressBar(l,l,folder,length = 30)







with open(folder+'/000.csv','r') as f:
    line = f.readline()
line0 = line.replace('\n','').split(',')
line0.append('frame')

files = os.listdir(folder)
files.sort()
arr = []
for i in range(1,len(files)):
     with open(folder+'/'+ files[i],'r') as f:
            
            for j,line in enumerate(f):
                if(j ==0):
                    continue
                row  =  line.replace('\n','').split(',')
                
                row.append(i)
                row[0] = int(row[0])
                row[1] = int(row[1])
                row[2] = int(row[2])
                
                row[3] = float(row[3])
                row[4] = float(row[4])
                row[5] = float(row[5])
                
                arr.append(row)
df = pd.DataFrame(data=arr,columns=line0)


# In[2]:


#df.groupby(['id','frame']).pipe(lambda group: group.)


df["angle2"] = (df["length"]-df["width"])/df.groupby(['id'])['length'].transform('max')
df["angle2"] = df["angle2"].apply(np.arccos)
df["angle2"] = df["angle2"].apply(np.rad2deg)


# In[3]:


df = df.loc[df["length"]*df["width"]>10]


# In[4]:


df3 = df.loc[(df["length"]*df["width"]>10) & (df["is_lost"]=='0')]
df3['frame_n'] = df3.groupby(['id'])['frame'].transform('count')


# In[5]:

if (format_output == 0 ):
    for i in list(df3.columns.values):
        df3[['id',i]].to_csv('/tmp/%s_%s.csv'%(output_name, i))
else:
    df3.to_csv(output_name)
print("==============================")
print("DONE")
print("==============================")

# df4 = df3.groupby(["id"])['frame_n',"angle2"].last()
# df4 = df4.loc[(df4['frame_n']>=30 )]
# # df2 = df3.groupby(["id"])['frame_n',"angle2"].last()
# df2 = df2.loc[(df2['frame_n']>=30 )]


# In[6]:


# import matplotlib.pyplot as plt
# fig = plt.figure(figsize=(10,7))

# ax =df3.angle2.plot(kind='kde',label=r"$EHEC$")
# #df4.angle2.plot(ax=ax,kind='kde',label=r"$m17$")
# ax.set_xlim(0,90)
# ax.set_xlabel(r"$\varphi^\circ$")
# ax.grid(linestyle=':')
# ax.legend()
# ax.set_title("Угол, при котором теряем частицу")
# # ax.set_yscale("log")
# plt.savefig("angle.png")

