from Classes import imagetomatrix as img
from Classes import datasplit as ds
from Classes import pcaalgorithm as pca
from Classes import knearestneighbors as knn
import pandas as pd
import numpy as np
"""
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.metrics import accuracy_score
"""

# Get the paths and labels of the data
train_percent = 80

yaleB = ds.MetaData(train_percent)

# Load the the paths as image matrices
image_width, image_height = 168, 192

training_set = img.ImageToMatrixClass(yaleB.training_paths, image_width, image_height)
testing_set = img.ImageToMatrixClass(yaleB.testing_paths, image_width, image_height)
train_matrix = training_set.matrix
test_matrix = testing_set.matrix
"""
# Probe with the ready-made functions
pca = PCA(n_components=0.99)
train = pca.fit_transform(train_matrix)
test = pca.transform(test_matrix)

print(train.shape, test.shape)

KNN = knn(8)
train_knn = KNN.fit(train,  yaleB.training_labels)
predicted_labels = KNN.predict(test)
score = accuracy_score(yaleB.testing_labels, predicted_labels)
print(score, "From the KNN function")
"""

# Do PCA dimension reduction
pca_images = pca.PCA(train_matrix, 99)
pca_model = pca_images.fit_svd()

train_tr = pca_images.transform(train_matrix)
test_tr = pca_images.transform(test_matrix)
print(train_tr.shape, test_tr.shape)

def k_accuracy (predicted_labels, testing_labels,k):
    print("this is the accuracy for k from 1 to 10:")
    print("k = ", k, ":")
    correct = 0
    accuracy = 0
    for i in range(predicted_labels.shape[0]):
        if predicted_labels[i] == testing_labels[i]:
            correct += 1
        else:
            continue
    accuracy = (correct / predicted_labels.shape[0]) * 100
    vector = vector_accuracy.insert(k, accuracy)
    print(accuracy)
    print(vector)
    #print("that is the new vector ", vector_accuracy1)
    #vector_k = ["k1", "k2", "k3", "k4", "k5", "k6", "k7"]
    #df = pd.DataFrame([accuracy], columns=['k1','k2', 'k3', 'k4', 'k5', 'k6', 'k7'])
    #print(df)
# Load KNN
vector_accuracy = []
for k in range (1,8):
    train_knn = knn.KNearestNeighbors(train_tr, yaleB.training_labels, k)
    predicted_labels = train_knn.predict(test_tr)

    k_accuracy(predicted_labels, yaleB.testing_labels, k)



# Check prediction
correct = 0
for i in range(predicted_labels.shape[0]):
    if predicted_labels[i] == yaleB.testing_labels[i]:
        correct += 1
    else:
        continue

print("That is the accuracy percent", (correct/predicted_labels.shape[0])*100)
accuracy = (correct/predicted_labels.shape[0])*100
"""
#acc = knn.k_accuracy(predicted_labels, yaleB.testing_labels)

#print("this is the accuracy for k from 1 to 10", acc)

# Loading KNN
train_knn = knn.KNearestNeighbors(train_tr, yaleB.training_labels)
distancess = train_knn._predict(test_tr[0])
print("that is the amount of the distances", len(distancess))