# Automatic tilt-shift photograpy
This propgram is used to apply the tilf-shift effect to a photo.

## Compile

### Environments

Install openCV

### Windows(Visual Studio)

1. Follow step 1-5 of the tutorial:
https://blogs.msdn.microsoft.com/microsoft_student_partners_in_taiwan/2016/05/14/%E5%B0%87opencv%E5%AE%8C%E7%BE%8E%E5%BB%BA%E7%BD%AE%E6%96%BCvisual-studio%E4%B8%8A/
2. Add all cpp files and header files to the project
3. 
4. Build

## Execute

Before running the program, set the command line in the properties window of the project, EX:0.5 500 50 test1.jpg. The program will show the level map and final tilf-shift result and save them as jpg files.

### Parameters
This program needs 4 parameters:
1. Sigma: smooth the image, affect the segment result, recommand value - 0.5
2. K: threshold value, affect the segment result, recommand value - 500
3. min_size: minimum component size, affect the segment result, recommand value - 50
4. image: the image path

### Focus Point
If you want to change the focus point, change the define value "focus_x" and "focus_y" in the code

## Citation
Image Segmentation

http://cs.brown.edu/people/pfelzens/segment/index.html

Belief propagation

http://cs.brown.edu/people/pfelzens/bp/index.html


## Author
Mei-Ling Chen ; Chia-Jung Chou ; Min-Tzu Wu
