import os
from skimage.feature import hog
from PIL import Image
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def extract_hog_features(image_path):
    image = Image.open(image_path).convert('L')
    image = np.array(image)
    
    hog_features = hog(image, orientations=9, pixels_per_cell=(8, 8),
                  cells_per_block=(2, 2), block_norm="L2-Hys")
    return hog_features

def load_features_and_labels(folder_path):
    features = []
    labels = []
    
    for image_name in sorted(os.listdir(folder_path)):
        if image_name.endswith('.tif'):
            label = image_name.split('_')[0]
            image_path = os.path.join(folder_path, image_name)
            try:
                hog_features = extract_hog_features(image_path)
                features.append(hog_features)
                labels.append(label)
            except Exception as e:
                print(f"Error processing {image_name}: {str(e)}")
    print("All images processed")
    return np.array(features), np.array(labels)

def save_results(results_dict, technique_name):
    results_dir = "./../final_results_svm"
    if not(os.path.exists(results_dir)):
        os.mkdir(results_dir)
    
    filename = os.path.join(results_dir, f"{technique_name}_results.txt")
    
    with open(filename, 'w') as f:
        f.write(f"Results for {technique_name}\n")
        f.write("="*50 + "\n\n")
        
        f.write("Dataset Information:\n")
        f.write("-"*20 + "\n")
        f.write(f"Number of samples: {results_dict['num_samples']}\n")
        f.write(f"Feature vector length: {results_dict['feature_length']}\n")
        f.write(f"Unique classes: {results_dict['unique_classes']}\n\n")
        
        f.write("Classification Results:\n")
        f.write("-"*20 + "\n")
        f.write(f"Accuracy: {results_dict['accuracy']}\n\n")
        f.write("Classification Report:\n")
        f.write(results_dict['classification_report'])
    
    print(f"\nResults saved to: {filename}")

import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion_matrices(all_cm, labels, technique_names):
    fig, axes = plt.subplots(2, 3, figsize=(15, 12)) 
    fig.suptitle('Confusion Matrices for All Techniques', fontsize=16)

    axes = axes.flatten()
    
    for i, (cm, ax, name) in enumerate(zip(all_cm, axes, technique_names)):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title(f'Confusion Matrix: {name}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  
    plt.show()


def train_and_evaluate_svm(features, labels, technique_name):
    results_dict = {
        'num_samples': len(features),
        'feature_length': features.shape[1],
        'unique_classes': np.unique(labels)
    }
    
    print(f"\nDataset info for {technique_name}:")
    print(f"Number of samples: {results_dict['num_samples']}")
    print(f"Feature vector length: {results_dict['feature_length']}")
    print(f"Unique classes: {results_dict['unique_classes']}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    # Train and evaluate
    svm_model = SVC(kernel='linear', C=1)
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)
    
    # Store results
    results_dict['accuracy'] = accuracy_score(y_test, y_pred)
    results_dict['classification_report'] = classification_report(y_test, y_pred)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Print results
    print(f"\nResults for {technique_name}:")
    print("Accuracy:", results_dict['accuracy'])
    print("Classification Report:\n", results_dict['classification_report'])
    
    # Save results to file
    save_results(results_dict, technique_name)
    
    return cm, results_dict
    

base_folder = "./../Output_Images"
techniques = ['original', 'agcwd_db1_b', 'global_db1_b', 'local_db1_b', 'poshe_db1_b', 'wthe_db1_b']

all_cm = []
technique_names = []

for technique in techniques:
    folder = os.path.join(base_folder, technique) if technique != 'original' else './../Datasets/DB1_B'
    print(f"\nProcessing images from: {folder}")
    features, labels = load_features_and_labels(folder)
    cm, results = train_and_evaluate_svm(features, labels, technique)
    all_cm.append(cm)
    technique_names.append(technique)

plot_confusion_matrices(all_cm, np.unique(labels), technique_names)
