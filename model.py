from tensorflow.keras.models import Sequential, load_model, clone_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import numpy as np
import os
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import KFold, train_test_split
import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout


def create_transfer_learning_model(input_shape):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze the base model

    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(3, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def load_test_data(directory):
    X_test = []
    y_test = []
    
    for file_name in os.listdir(directory):
        if file_name.endswith('.npz'):
            file_path = os.path.join(directory, file_name)
            data = np.load(file_path)
            
            image = data['image']
            if len(image.shape) == 3:
                image = np.expand_dims(image, axis=0)
                
            label = data['label']
            if len(label.shape) == 1:
                label = np.expand_dims(label, axis=0)
                
            X_test.append(image)
            y_test.append(label)
    
    X_test = np.concatenate(X_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)
    
    return X_test, y_test

def load_data_from_directory(directory):
    X_data = []
    y_data = []
    
    for file_name in os.listdir(directory):
        if file_name.endswith('.npz'):
            file_path = os.path.join(directory, file_name)
            data = np.load(file_path)
            
            # Add a batch dimension to the image if needed
            image = data['image']
            if len(image.shape) == 3:
                image = np.expand_dims(image, axis=0)  # Make it (1, 240, 240, 3)
                
            # Add a batch dimension to the label if needed
            label = data['label']
            if len(label.shape) == 1:
                label = np.expand_dims(label, axis=0)  # Make it (1, 3)
                
            X_data.append(image)
            y_data.append(label)
            
            # print(f"Loaded {file_name}: X shape = {image.shape}, y shape = {label.shape}")

    # Convert lists to numpy arrays
    X_data = np.concatenate(X_data, axis=0)
    y_data = np.concatenate(y_data, axis=0)
    
    print(f"Final shapes - X: {X_data.shape}, y: {y_data.shape}")
    return X_data, y_data

def create_model(input_shape):
    model = Sequential([
        # First Conv Block
        Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        Dropout(0.25),  # Added dropout after first maxpool
        
        # Second Conv Block
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        # Third Conv Block
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),  # Added dropout after last maxpool
        
        # Fourth Conv Block
        Conv2D(128, (3, 3), activation='relu'),
        
        # Flatten and Dense layers
        Flatten(),
        Dense(64, activation='relu'),
        Dense(3, activation='softmax')  # 3 classes
    ])
    
    return model

# Prepare the data
directories = ['processed_data_GLI', 'processed_data_MET', 'processed_data_MEN']  # List of directories containing .npz files

X_data_list = []
y_data_list = []

for directory in directories:
    X_data, y_data = load_data_from_directory(directory)
    X_data_list.append(X_data)
    y_data_list.append(y_data)

# Concatenate data from all directories
X_data = np.concatenate(X_data_list, axis=0)
y_data = np.concatenate(y_data_list, axis=0)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.15, random_state=42)

model = create_transfer_learning_model(input_shape=(240, 240, 3))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Create data augmentation generator
datagen = ImageDataGenerator(
    rotation_range=20,
    horizontal_flip=True,
    vertical_flip=True
)

# Fit the generator on the training data
datagen.fit(X_train)

# Callbacks
early_stopping = EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)

# Train the model
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_val, y_val),  # Use the validation data directly
    epochs=50,
    callbacks=[early_stopping, lr_scheduler]
)

# Save the model
model.save('final_model.keras')


# Load test data
directories = ['test_processed_data_GLI', 'test_processed_data_MET', 'test_processed_data_MEN']  # List of directories containing .npz files

X_test_list = []
y_test_list = []

for directory in directories:
    X_test, y_test = load_test_data(directory)
    X_test_list.append(X_test)
    y_test_list.append(y_test)

# Concatenate data from all directories
X_test = np.concatenate(X_test_list, axis=0)
y_test = np.concatenate(y_test_list, axis=0)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f"Test accuracy: {test_accuracy:.4f}, Test loss: {test_loss:.4f}")

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Get the predicted class indices
y_true_classes = np.argmax(y_test, axis=1)  # Get the true class indices

# Generate classification report
report = classification_report(y_true_classes, y_pred_classes, target_names=['GLI', 'MEN', 'MET'])
print(report)

# Save the test results to training_history.json
with open('training_history.json', 'w') as f:
    json.dump(history.history, f)




# Initialize K-fold cross validation
# n_folds = 5
# kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# # Lists to store the performance metrics for each fold
# fold_scores = []
# histories = []

# # Perform k-fold cross validation
# for fold, (train_idx, val_idx) in enumerate(kfold.split(X_data)):
#     print(f'\nFold {fold + 1}/{n_folds}')
    
#     # Split the data
#     X_train, X_val = X_data[train_idx], X_data[val_idx]
#     y_train, y_val = y_data[train_idx], y_data[val_idx]
    
#     # Create and compile the model
#     model = create_model(input_shape=(240, 240, 3))
#     model.compile(
#         optimizer='adam',
#         loss='categorical_crossentropy',
#         metrics=['accuracy']
#     )
    
#     # Create data augmentation generator for training data
#     datagen = ImageDataGenerator(
#         rotation_range=20,
#         horizontal_flip=True,
#         vertical_flip=True
#     )
    
#     # Fit the generator on the training data
#     datagen.fit(X_train)
    
#     # Callbacks
#     early_stopping = EarlyStopping(
#         monitor='val_accuracy',
#         patience=50,
#         restore_best_weights=True
#     )
    
#     lr_scheduler = ReduceLROnPlateau(
#         monitor='val_loss',
#         factor=0.1,
#         patience=5,
#         min_lr=1e-6
#     )
    
#     # Train the model
#     history = model.fit(
#         datagen.flow(X_train, y_train, batch_size=32),
#         validation_data=(X_val, y_val),
#         epochs=50,
#         callbacks=[early_stopping, lr_scheduler]
#     )
    
#     # Evaluate the model
#     scores = model.evaluate(X_val, y_val, verbose=0)
#     print(f'Fold {fold + 1} - Loss: {scores[0]:.4f}, Accuracy: {scores[1]:.4f}')
    
#     # Store the scores and history
#     fold_scores.append(scores[1])  # storing accuracy
#     histories.append(history.history)
    
#     # Save the model for this fold
#     model.save(f'model_fold_{fold + 1}.keras')

# # Print the results
# print('\nK-Fold Cross-Validation Results:')
# print(f'Mean Accuracy: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})')

# # Save the cross-validation results
# cv_results = {
#     'fold_scores': fold_scores,
#     'histories': histories,
#     'mean_accuracy': float(np.mean(fold_scores)),
#     'std_accuracy': float(np.std(fold_scores))
# }

# with open('cv_results.json', 'w') as f:
#     json.dump(cv_results, f)

# # Assuming you have already trained your models and stored them
# fold_models = []  # List to store models from each fold
# fold_scores = []  # List to store scores for each fold

# # After training each fold, append the model to fold_models
# for fold in range(n_folds):
#     model = load_model(f'model_fold_{fold + 1}.keras')  # Load each fold's model
#     fold_models.append(model)

# # Average the weights of the models
# def average_models(models):
#     # Get the weights of the first model
#     avg_weights = [np.zeros_like(w) for w in models[0].get_weights()]
    
#     # Sum the weights of all models
#     for model in models:
#         for i, w in enumerate(model.get_weights()):
#             avg_weights[i] += w
    
#     # Average the weights
#     avg_weights = [w / len(models) for w in avg_weights]
    
#     # Create a new model instance
#     avg_model = clone_model(models[0])  # Create a new model with the same architecture
#     avg_model.build(models[0].input_shape)  # Build the model with the input shape
#     avg_model.set_weights(avg_weights)  # Set the averaged weights
    
#     return avg_model

# # Create the averaged model
# averaged_model = average_models(fold_models)

# # Compile the averaged model
# averaged_model.compile(
#     optimizer='adam',  # You can choose any optimizer you prefer
#     loss='categorical_crossentropy',  # Use the same loss function as before
#     metrics=['accuracy']  # Metrics to monitor
# )








# # Evaluate each fold's model and the averaged model
# best_model = None
# best_accuracy = 0.0

# for fold, model in enumerate(fold_models):
#     test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
#     print(f"Fold {fold + 1} - Test accuracy: {test_accuracy:.4f}, Test loss: {test_loss:.4f}")
    
#     # Save the best model based on test accuracy
#     if test_accuracy > best_accuracy:
#         best_accuracy = test_accuracy
#         best_model = model

# # Evaluate the averaged model
# avg_test_loss, avg_test_accuracy = averaged_model.evaluate(X_test, y_test, verbose=2)
# print(f"Averaged Model - Test accuracy: {avg_test_accuracy:.4f}, Test loss: {avg_test_loss:.4f}")

# # Check if the averaged model is the best
# if avg_test_accuracy > best_accuracy:
#     best_accuracy = avg_test_accuracy
#     best_model = averaged_model

# # Save the best model
# best_model.save('best_model.keras')
# print(f"Best model saved with accuracy: {best_accuracy:.4f}")

