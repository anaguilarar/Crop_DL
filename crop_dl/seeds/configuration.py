import random

class RiceDetectionConfig:
    
    
    model_weights_path = "model_weights/maskrcnn_rice_seeds_20230307-161014/"
    new_image_path = "rice_images/new_images"
    new_images_training_path = "rice_images/new_training"
    training_path = "rice_images/rice_seeds/training"
    images_path = "rice_images/rice_seeds"
    
    random_transforms_params = {
        'zoom': random.choice([
            random.randint(-85, -65),
            random.randint(20, 50)]),
        'clahe': random.randint(10,70),
        'hsv': [random.randint(10,100),
                random.randint(0,40),
                random.randint(0,50)]
        }

