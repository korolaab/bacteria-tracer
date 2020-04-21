#!/bin/bash
folder=$1
x=$2
y=$3
n=0
for file in `find $folder | sort`
    do
        convert $file -format "%[pixel:p{$x,$y}]\n" info: | sed s/[^0-9]//g >> /tmp/pixel_${x}x${y}.txt
        fname=` printf "orig_crop_%03d.png" $n`
        echo $fname
        convert $file -colorspace sRGB -crop 100x100+${x-50}+${y-50} -resize 300x300 /tmp/$fname
        fname2=` printf "mark_crop_%03d.png" $n`
        echo $fname2
        convert /tmp/$fname -colorspace sRGB -fill red -draw "circle 148,148,152,152" /tmp/$fname2
        n=$((n+1))
    done        
echo "making plot"
./make_plot.py /tmp/pixel_${x}x${y}.txt 

for ((i = 0;i<n;i++))
    do
        
        fname=` printf "orig_crop_%03d.png" $i`
        fname2=` printf "mark_crop_%03d.png" $i` 
        fname3=` printf "cache_plot_%03d.png" $i`
        fname_out=` printf "output_%03d.png" $i`
        echo $fname_out
        montage  -mode concatenate /tmp/$fname /tmp/$fname2 /tmp/$fname3 -tile 1x3 /tmp/$fname_out
    done
ffmpeg -framerate 3 -i /tmp/output_%03d.png output_${x}x${y}.mp4
rm /tmp/orig_crop*.png
rm /tmp/mark_crop*.png
rm /tmp/cache_plot*.png
rm /tmp/output_*.png  
rm /tmp/pixel_${x}x${y}.txt
