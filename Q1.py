# %%
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import sklearn.svm as svm
from skimage.filters import gabor_kernel , gabor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import numpy as np

# %%
def subsample(X, y, samples_per_label):
    indices = []

    for label in np.unique(y):
        label_indices = np.where(y == label)[0]
        rand_generator = np.random.default_rng(seed=42)
        selected = rand_generator.choice(label_indices, 
                                  samples_per_label,
                                  replace=False)
        indices.extend(selected)
    
    indices = np.array(indices)
    
    return X[indices], y[indices]

def process_image(img):
    reshaped_img = img.reshape((28 ,28))
    reshaped_img = reshaped_img.astype(np.uint8) # fixes error with resize

    downsampled_img = cv2.resize(reshaped_img, (14 ,14))
    downsampled_img = downsampled_img / 255.0 # normalize pixel values
    return downsampled_img

# %%
ds = fetch_openml('mnist_784', as_frame = False )

sample_X, sample_y = subsample(ds.data, ds.target, 1000) # want 1000 examples per label
sample_y = sample_y.astype(int)
processed_X = np.apply_along_axis(process_image, 1, sample_X)
processed_X = processed_X.reshape(processed_X.shape[0], -1) # flatten for SVM


x_train, x_test, y_train, y_test = train_test_split(processed_X, sample_y, 
                                         test_size =0.2, random_state =42)


print("Full:")
print(ds.target.size)
print(np.unique(ds.target, return_counts=True))

print("Sampled:")
print(sample_y.size)
print(np.unique(sample_y, return_counts=True))

# %%
classifier = svm.SVC(C=1.0, kernel='rbf', gamma='auto')
classifier.fit(x_train, y_train)


# %%
y_pred = classifier.predict(x_test)
#print(np.unique(y_pred, return_counts=True))
print("classification accuracy before tuning:", classifier.score(x_test, y_test))
support_sample_ratio = classifier.n_support_.sum() / len(x_train)
print("ratio of support samples to total training samples:", support_sample_ratio)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
#print(cm)

# For better visualization:
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('10-Class Confusion Matrix')
plt.show()

# %%
params = {
    "C":[0.01, 0.1, 1.0, 10, 100, 1000]
}

grid_search = GridSearchCV(
    estimator=classifier,
    param_grid=params,
    cv=5, 
    scoring='accuracy',
    verbose=1, 
)

# perform grid search on data pre-split
grid_search.fit(processed_X, sample_y)

results = grid_search.cv_results_
for i in range(len(results['params'])):
    params = results['params'][i]
    mean_score = results['mean_test_score'][i]
    std_score = results['std_test_score'][i]
    print(f"C = {params['C']:8.1f} | Validation Accuracy: {mean_score:.4f} | Standard Deviation: {std_score:.4f}")

# %%

# all filter parameter options
thetas = np.arange(0, np.pi, np.pi/4) 
frequencies = np.arange(0.05, 0.5, 0.15)  
bandwidths = np.arange(0.3, 1, 0.3) 

# Create subplot grid (6x6 for 36 filters)
fig, axes = plt.subplots(6, 6, figsize=(12, 12))
axes = axes.ravel()

filter_idx = 0
for f in frequencies:
    for t in thetas:
        for b in bandwidths:
            
            kernel = gabor_kernel(frequency=f, theta=t, bandwidth=b)
            
            axes[filter_idx].imshow(kernel.real, cmap='gray')
            axes[filter_idx].axis('off')
            
            filter_idx += 1

# gk = gabor_kernel(frequency =freq, theta =theta, bandwidth = bandwidth)
# plt.figure(1); plt.clf(); plt.imshow(gk.real)
# plt.figure(2); plt.clf(); plt.imshow(gk.imag)

# image = x_train[0].reshape((14 ,14))
# coeff_real , _ = gabor(image , frequency=freq, theta=theta,
#                        bandwidth = bandwidth)

# plt.figure(1); plt.clf(); plt.imshow(coeff_real)

# %%
# smaller train and test set for gabor to compensate for runtime
gabor_sample_X, gabor_sample_y = subsample(ds.data, ds.target, 200)
gabor_sample_y = gabor_sample_y.astype(int)
processed_X = np.apply_along_axis(process_image, 1, gabor_sample_X)
processed_X = processed_X.reshape(processed_X.shape[0], -1) # flatten for SVM


gabor_train_X, gabor_validation_X, gabor_train_y, gabor_validation_y = train_test_split(
    processed_X,gabor_sample_y,
    test_size=0.5, 
    random_state=42,
    stratify=gabor_sample_y  # Ensures 100 per label in both
)

thetas = np.arange(0, np.pi, np.pi/4) 
frequencies = np.arange(0.05, 0.5, 0.15)  
bandwidths = np.arange(0.3, 1, 0.3) 

def get_gabor_coefficients(images, thetas, frequencies, bandwidths):
    num_samples = images.shape[0]
    num_filters = len(thetas) * len(frequencies) * len(bandwidths)
    gabor_coeffs = np.zeros((num_samples, 196 * num_filters)) # since 14 x 14 = 196
    
    for image_id in range(num_samples):
        img = images[image_id].reshape((14,14)) # reconstruct downsampled image
        
        curr_filter = 0 # iterate through filter bank
        for f in frequencies:
            for t in thetas:
                for b in bandwidths:
                    coeff_real , _ = gabor(img , frequency=f, theta=t,
                        bandwidth = b)
                    start_coeff_index = 196 * curr_filter
                    end_coeff_index = 196 * (curr_filter + 1)
                    gabor_coeffs[image_id, start_coeff_index:end_coeff_index] = coeff_real.flatten()
                    curr_filter += 1
                    
    return gabor_coeffs

x_train_gabor_coeffs = get_gabor_coefficients(gabor_train_X, thetas, frequencies, bandwidths)
x_test_gabor_coeffs = get_gabor_coefficients(gabor_validation_X, thetas, frequencies, bandwidths)

# %%
gabor_classifier = svm.SVC(C=1000.0, kernel='rbf', gamma='auto')
gabor_classifier.fit(x_train_gabor_coeffs, gabor_train_y)

gabor_y_train_pred = gabor_classifier.predict(x_train_gabor_coeffs)
train_accuracy = accuracy_score(gabor_train_y, gabor_y_train_pred)
print(f"Training Accuracy: {train_accuracy}")

gabor_y_test_pred = gabor_classifier.predict(x_test_gabor_coeffs)
test_accuracy = accuracy_score(gabor_validation_y, gabor_y_test_pred)
print(f"Validation Accuracy: {test_accuracy}")

# %%
# expand size, can't do by much because compute scales too intensely
larger_thetas = np.arange(0, np.pi, np.pi/5) # 5 rotations
larger_frequencies = np.arange(0.05, 0.5, 0.15)  # 3 frequencies
larger_bandwidths = np.arange(0.1, 1, 0.3) # 3 bandwiths

x_train_larger_gabor_coeffs = get_gabor_coefficients(gabor_train_X, larger_thetas, larger_frequencies, larger_bandwidths)
x_test_larger_gabor_coeffs = get_gabor_coefficients(gabor_validation_X, larger_thetas, larger_frequencies, larger_bandwidths)



# %%
#using PCA with 1000 features per img instead of 196 * 96 = 18816
pca = PCA(n_components=20)
gabor_x_train_pca = pca.fit_transform(x_train_larger_gabor_coeffs)
gabor_x_test_pca = pca.transform(x_test_larger_gabor_coeffs)

scaler = StandardScaler()
gabor_x_train_scaled = scaler.fit_transform(gabor_x_train_pca)
gabor_x_test_scaled = scaler.transform(gabor_x_test_pca)

# Use scaled data for SVM:
gabor_classifier = svm.SVC(C=1, kernel='rbf', gamma='auto')
gabor_classifier.fit(gabor_x_train_scaled, gabor_train_y)

larger_gabor_y_train_pred = gabor_classifier.predict(gabor_x_train_scaled)
larger_gabor_y_test_pred = gabor_classifier.predict(gabor_x_test_scaled)


train_accuracy = accuracy_score(gabor_train_y, larger_gabor_y_train_pred)
print(f"Training Accuracy: {train_accuracy}")

test_accuracy = accuracy_score(gabor_validation_y, larger_gabor_y_test_pred)
print(f"Validation Accuracy: {test_accuracy}")

# %% [markdown]
# 


