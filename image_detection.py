import sys
from transformers import pipeline
from PIL import Image, ImageDraw

checkpoint = "google/owlv2-base-patch16-ensemble"
detector = pipeline(model=checkpoint, task="zero-shot-object-detection")

filename = sys.argv[1]
image = Image.open(filename)


# The list of words must be separated by single spaces
labels = input("Enter the items you want to detect: ").split(" ")

# Run the pipeline and get results
predictions = detector(
     image,
     candidate_labels=labels,
 )

  # Move the drawing logic inside the loop to handle individual predictions
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

  # Save as a unique image with a unique name for each detection pass
image.save(f"{filename.split('.')[0]}_detection.png")