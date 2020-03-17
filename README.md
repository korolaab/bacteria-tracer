# bacteria-tracer
Coordinates of bacteria from video file
## Requirements
- python 3.6.9
- TensorFlow 1.14.0
- OpenCV 4.1.2
- Keras 2.3.1
- NumPy 1.18.1
## How to use
```
$ find_coordinates --input={video filename} --output={outpu json file}
```
If you want to get video with marked bacteria
```
$ find_coordinates --input={video filename} --output={output json file}\
                    > -- output_video={ouput video filename}
```
## JSON file format
```
{
    "video_filename": video filename,
    "frames": {
                "0":[[x1,y1],[x2,y2],...,[xn,yn]],
                "1":[[x1,y1],[x2,y2],...,[xn,yn]],
                ---------------------------------
                "m":[[x1,y1],[x2,y2],...,[xn,yn]]
        }
}
```
[x,y] is coordinate of one bacteria on frame
"0" - "m" is number of frame
## Example
Example of proccesed [video](https://youtu.be/4pJVfj9q65I).
