Action Classifier in Video using Convolutional Neural Network and LSTMs
on sports data.

## Architecture
2-Stream CNN approach was followed with one CNN taking resized RGB images
to the other taking optical flow generated from the image as part of preprocessing.

The RGB CNN learns object detection whereas the optical flow CNN learns
motion in the frames.

## Preprocessing
 - All the frames were extracted from the video and save as RGB images.
 - Separate frames were generate by calculating the optical flow.
 - The frames were resized to 64x64x3.
 - As the videos were of unequal length, we split the frames in equal 
    20 bins and takes average of frames in a particular bin and then stack
    them over one another to finally form a 20x64x64x3 dimension video data.
    
## Results
 - The training accuracy was 87% while the validation accuracy was 80%.
  

