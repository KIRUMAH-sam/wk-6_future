# generate_dataset.py
import os
from PIL import Image, ImageDraw, ImageFont
import random

OUT = "recyclables_dataset"
CLASSES = ["bottle", "can", "paper"]
IMG_SIZE = (128, 128)
IMAGES_PER_CLASS = 100

os.makedirs(OUT, exist_ok=True)
for cls in CLASSES:
    d = os.path.join(OUT, cls)
    os.makedirs(d, exist_ok=True)

def draw_bottle(draw):
    w, h = IMG_SIZE
    # bottle body
    draw.ellipse([(w*0.35, h*0.2), (w*0.65, h*0.8)], outline="black", width=3, fill=(180,220,255))
    # neck
    draw.rectangle([(w*0.45, h*0.05),(w*0.55, h*0.2)], fill=(180,220,255), outline="black", width=3)

def draw_can(draw):
    w, h = IMG_SIZE
    draw.rectangle([(w*0.3, h*0.2),(w*0.7,h*0.75)], outline="black", width=3, fill=(220,220,180))
    # can top
    draw.ellipse([(w*0.28,h*0.15),(w*0.72,h*0.25)], outline="black", width=2, fill=(200,200,160))

def draw_paper(draw):
    w, h = IMG_SIZE
    # a crumpled rectangle simulated by polygon
    jitter = lambda x: x + random.randint(-6,6)
    poly = [(jitter(int(w*0.25)), jitter(int(h*0.25))),
            (jitter(int(w*0.75)), jitter(int(h*0.2))),
            (jitter(int(w*0.8)), jitter(int(h*0.7))),
            (jitter(int(w*0.3)), jitter(int(h*0.75)))]
    draw.polygon(poly, fill=(255,255,230), outline="black")

for cls in CLASSES:
    for i in range(IMAGES_PER_CLASS):
        img = Image.new("RGB", IMG_SIZE, (255,255,255))
        draw = ImageDraw.Draw(img)
        # background noise
        if random.random() < 0.3:
            for _ in range(10):
                x0 = random.randint(0, IMG_SIZE[0])
                y0 = random.randint(0, IMG_SIZE[1])
                x1 = x0 + random.randint(1,10)
                y1 = y0 + random.randint(1,10)
                draw.ellipse((x0,y0,x1,y1), fill=(240,240,240))
        # draw item
        if cls == "bottle":
            draw_bottle(draw)
        elif cls == "can":
            draw_can(draw)
        else:
            draw_paper(draw)
        # rotation and jitter
        if random.random() < 0.6:
            img = img.rotate(random.uniform(-25,25), expand=False, fillcolor=(255,255,255))
        filename = os.path.join(OUT, cls, f"{cls}_{i:03d}.png")
        img.save(filename)
print("Dataset created in", OUT)
