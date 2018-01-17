import random

def real_and_models():
    
    real = int(random.random() < 0.5)
    
    true_prediction1 = 0.5 * real + 0.5 * random.random()
    true_prediction2 = 0.5 * real + 0.5 * random.random()
    
    first_right = int(random.random() < 0.55)
    second_right = int(random.random() < 0.6)
    
    return {
        'real' : real,
        'first_model' : first_right and true_prediction1 or 1 - true_prediction1,
        'second_model' : second_right and true_prediction2 or 1 - true_prediction2
    }