import os
from pathlib import Path

import requests
from PIL import Image
from io import BytesIO

from fastai.vision import load_learner, open_image


model_path = Path(
    os.path.join(
        os.path.abspath(os.path.dirname(__name__)),
        'ml',
        'models'
    )
)
comics_learner = load_learner(path=model_path, file="comics.pkl")
classes = comics_learner.data.classes

def get_image_by_url(url):
    """ Load img from url and save to path/filename locally """
    resp = requests.get(url)
    
    if resp.status_code == 200:
        # need convert to RGB to save in .jpeg further
        return open_image(BytesIO(resp.content))


def predict_comics(img):
	"""
	Returns:
		- predicted class name
		- dict with class_name: predict_proba
	"""
	yhat_class, _, yhat_probas = comics_learner.predict(img)
	yhat_class = str(yhat_class)
	yhat_probas = list((yhat_probas.numpy() * 100).round(2))
	
	return yhat_class, dict(zip(classes, yhat_probas))