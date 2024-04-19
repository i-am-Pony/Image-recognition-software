# Image-recognition-software-WIP
A script that will scan an image, check for and detect a specific object, or list of items, and returns if they are present, their likelihood as a %, make a red box around it. Then save the results as a separate iage file with a unique name.
##
We used this script to detect if there were any pedestrians in a given crosswalk. This can be further developed as a safety software aimed at keepinng people safe from traffic accidents.

1) Load a virtual environment. Like:
```bash
python -m venv venv/
``` 
or python3 -m venv venv/
whatever your version.
##

2) install dependencies
a) transformers
```bash
pip install transformers
```
b) PIL (Pillow)
```bash
pip3 install Pillow
```
c) Instal SciPy
```bash
pip3 install scipy
```
##
3) Create a python file. Nameit <YOUR_FILE>.py
 
##
4) Import the needed libraries
```bash
import sys
from transformers import pipeline
from PIL import Image, ImageDraw
```
##
5) Import the model and define the pipeline
```bash
checkpoint = "google/owlv2-base-patch16-ensemble"
detector = pipeline(model=checkpoint, task="zero-shot-object-detection")
```
##
6) Load the image passed as argument to the script
```bash
filename = sys.argv[1]
image = Image.open(filename)
```
##
7) Receive the list of objects to detect as user input
```bash
# The list of words must be separated by single spaces
labels = input("Enter the items you want to detect: ").split(" ")
```
##
8) Run the pipeline and get results
```bash
predictions = detector(
     image,
     candidate_labels=labels,
 )
```
##
9) Draw a boxe(s) around the desired object(s). Predict the probability of each object as a %
```bash
 i=1
draw = ImageDraw.Draw(image)
for prediction in predictions:
    box = prediction["box"]
    label = prediction["label"]
    score = prediction["score"] * 100
    suffix = ''
    if 11 <= (i % 100) <= 13:
        suffix = 'th'
    else:
        suffix = ['th', 'st', 'nd', 'rd', 'th'][min(i % 10, 4)]
    print(f"The word {label} is the {i}{suffix} most related to the image with a confidence of {score:.2f}%")
    i+=1

      xmin, ymin, xmax, ymax = box.values()
    draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
    draw.text((xmin, ymin), f"{label}: {round(score,2)}", fill="black")
```
##
10) Save the newly created image with a new name
```bash
image.save(f"{filename.split('.')[0]}_{object_to_detect}_detection.png")
```
##
11) Run the script with the image file using the file you saved earlier "<YOUR_FILE>.py"
```bash
python <YOUR_FILE>.py <IMAGE_TO_INSPECT>.png
```
# The module should return "Enter the items you want to detect:" Enter the object(s) you want to detect example "people"
