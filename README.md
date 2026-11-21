# Image Classification Web App

Flask + TensorFlow/Keras CNN trained on CIFAR-10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).
Upload or drag in an image, or switch to the live camera for real-time predictions.

## Setup
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Train the model (downloads the dataset automatically, ~170 MB)
```bash
python train.py --epochs 30
```
Expect roughly 85-90% test accuracy. A GPU is faster, but CPU works (about 1-2 minutes per epoch).
Outputs go to `models/` (model, class names, metrics, training curves) and 20 test images go to `samples/`.

## 2. Run the web app
```bash
python app.py
```
Open http://127.0.0.1:5000. Try the pictures in `samples/`.

## API
`POST /api/predict` with a multipart field `image` returns
`{"predictions": [{"label": "cat", "confidence": 0.93}, ...]}` (top 5).

```bash
curl -F image=@samples/00_cat.png http://127.0.0.1:5000/api/predict
```

## Notes
- The model only knows the 10 CIFAR-10 classes, so anything else gets mapped to the closest one.
- Images are centre-cropped to a square and resized to 32x32 before prediction.
- The browser only allows camera access on localhost or HTTPS.
