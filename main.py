#IMAGE TO TEXT
#AI for this: Salesforce/blip-image-captioning-large

#pip install pillow
#pip install transformers torch torchvision pillow
#python.exe -m pip install --upgrade pip
#pip install transformers pillow

# Use a pipeline as a high-level helper
from transformers import pipeline
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = AutoModelForVision2Seq.from_pretrained("Salesforce/blip-image-captioning-large")

image = Image.open("Images/noveny1.jpg")

# Kulcs: ilyen prompttal kényszerítjük, hogy fajnevet próbáljon mondani
prompt = "a photo of a plant species:"

inputs = processor(images=image, text=prompt, return_tensors="pt")

generated_ids = model.generate(**inputs, max_new_tokens=50)
caption = processor.decode(generated_ids[0], skip_special_tokens=True)

print("BLIP növényfelismerés:", caption)