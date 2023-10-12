import os
import cv2
import sys
import numpy as np
from imutils import paths
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import RocCurveDisplay, roc_curve, auc
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import load_model

dir_path = sys.argv[2]

img_paths = list(paths.list_images(dir_path))

model = load_model(sys.argv[1])
labels = ['healthy', 'partially injured', 'fully_ruptured']
y_true = []
y_pred = []
pred_ = []
pred_score = []


for img_path in img_paths:
    print(img_path)
    y_ = int(img_path.split(os.path.sep)[-2])
    y_true.append(y_)
    
    img = cv2.imread(img_path)
    h, w, c = img.shape

    crop_size = min(h, w)
    c_x = w // 2
    c_y = h // 2
    
    xmin = c_x - crop_size // 2
    xmax = c_x + crop_size // 2
    ymin = c_y - crop_size // 2
    ymax = c_y + crop_size // 2
    
    center_crop = img[ymin:ymax, xmin:xmax, :] 
    
    img_rgb = cv2.cvtColor(center_crop, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (224, 224))

    img_rgb = np.asarray(img_rgb, dtype=np.float32).reshape(1, 224, 224, 3)
    img_rgb = (img_rgb / 127.5) - 1

    pred = model.predict(img_rgb)[0]
    pred_.append(pred)
    pred_score.append(max(pred))

    idx = np.argmax(pred)
    y_pred.append(idx)

    label = labels[idx]
    print(label, ' ', pred[idx])

print(y_true)
print(y_pred)
print(len(y_true), len(y_pred))
print(classification_report(y_true, y_pred, target_names=labels))
print(roc_auc_score(y_true, pred_, multi_class='ovr'))
print(pred_)
pred_ = np.array(pred_)
#print(len(pred_))
print(pred_.shape)

label_binarizer = LabelBinarizer().fit(y_true)
y_true_onehot = label_binarizer.transform(y_true)

fig, ax = plt.subplots(figsize=(6, 6))
RocCurveDisplay.from_predictions(y_true_onehot[:, 0],
                                 pred_[:, 0],
                                 name="Class 0 vs the rest",
                                 color="red",
                                 ax=ax
                                 )

RocCurveDisplay.from_predictions(y_true_onehot[:, 1],
                                 pred_[:, 1],
                                 name="Class 1 vs the rest",
                                 color="green",
                                 ax=ax
                                 )

RocCurveDisplay.from_predictions(y_true_onehot[:, 2],
                                 pred_[:, 2],
                                 name="Class 2 vs the rest",
                                 color="blue",
                                 ax=ax
                                 )                                 

plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curve (One-vs-Rest)")
plt.legend()
plt.savefig('roc.png')

cm = confusion_matrix(y_true, y_pred)
cm_display = ConfusionMatrixDisplay(cm).plot()
plt.savefig('confusion_matrix.png')
