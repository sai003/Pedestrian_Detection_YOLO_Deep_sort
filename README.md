# Pedestrian_Detection_YOLO_Deep_sort

We have not submitted any images as part of the code.
Please copy the actual training (and panoptic images) and test dataset images
to the folders mentioned below before running the code.
Note that in the code, we will run the model on only one of the datasets at a time.
Please refer to the code if you have any doubts.

INPUT_IMAGES_FOLDER = "./step_images/STEP-ICCV21-02/"
MASK_IMAGES_FOLDER = "./step_images/0002/"

INPUT_IMAGES_FOLDER = "./step_images/test/STEP-ICCV21-07/"
MASK_IMAGES_FOLDER = "./step_images/test/STEP-ICCV21-07/"

INPUT_IMAGES_FOLDER = "./step_images/STEP-ICCV21-09/"
MASK_IMAGES_FOLDER = "./step_images/0009/"

# all the output images will be stored in this location

OUTPUT_IMAGES_FOLDER = "./output/"

Also, note that our system will first process all the image files
and then it will show the output images in a loop.

# Run the code

Open Terminal and proceed to the "Pedestrian_Detection_YOLO_Deep_sort" folder
and then install all the dependencies by running: "pip install -r requirements.txt"
and then run the code by: "python main.py"

# Instructions

While the output video is running, please use "P" button on your keyboard to pause or unpause the video
The drawing of user given box can be done while the video is paused.
Please press the button "Q" on your keyboard to terminate the output video.
