import albumentations as A
import cv2
import os
import csv
image_path="/home/hoola/code/MobileNetLiveliness/data/interim/FaceDataset/Fake"
destination_path="/home/hoola/code/MobileNetLiveliness/data/interim/FaceDataset/Albumenated"

# Declare an augmentation pipeline
transform = A.ReplayCompose([
    A.OneOf([
        A.GlassBlur(max_delta=2,p=0.3),
        A.RandomContrast(p=0.5),
        A.JpegCompression(p=0.25),
        A.HueSaturationValue(p=0.25),
        A.RandomBrightness(p=0.25),
        A.MotionBlur(p=0.25),
        A.RandomFog(p=0.25),
        A.RandomSunFlare(num_flare_circles_lower=1, num_flare_circles_upper=3, src_radius=50,p=0.25),
        A.RandomGamma(p=0.25),
    ], 0.75)
])

def get_applied_transforms(transforms):
    applied = {}
    for transform in transforms:
        applied[transform['__class_fullname__']] = transform["applied"]
    
    return applied
for images in os.listdir(image_path):
    file=os.path.join(image_path,images)
    print(file)
    image=cv2.imread(file)
    print(image)
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    image_name=str(file.split('/')[-1])
    label=transformed["replay"]["transforms"][0]["transforms"]

    print(get_applied_transforms(label))
    csv_columns = ['Image','Details']
    dict_data = [
    {"Image": image_name, 'Details':str(get_applied_transforms(label))},
    ]
    csv_file = "fake.csv"
    try:
        with open(csv_file, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in dict_data:
                writer.writerow(data)
    except IOError:
        print("I/O error")
    dst=os.path.join(destination_path,image_name)
    transformed_image= cv2.imwrite(dst,transformed_image)
