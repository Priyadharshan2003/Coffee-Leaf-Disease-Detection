import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

def create_model(input_shape=(224, 224, 3), num_classes=4):
    """
    Create a fine-tuned ResNet50 model for coffee leaf disease classification
    
    Args:
        input_shape: Input image dimensions (height, width, channels)
        num_classes: Number of disease classes to predict
        
    Returns:
        Compiled Keras model
    """
    # Load pre-trained ResNet50 model without the top classification layer
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom classification layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Create the full model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def plot_training_history(history):
    """
    Plot training and validation accuracy/loss
    
    Args:
        history: History object returned by model.fit()
    """
    # Create directory for saving plots
    os.makedirs('plots', exist_ok=True)
    
    # Plot training & validation accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('plots/training_history.png')
    plt.close()

def train_model(args):
    """
    Train the coffee leaf disease detection model
    
    Args:
        args: Command line arguments
    """
    # Define image dimensions and preprocessing
    img_width, img_height = 224, 224
    batch_size = args.batch_size
    
    # Create data generators with augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    test_datagen = ImageDataGenerator(rescale=1.0/255)
    
    # Load training and validation data
    train_generator = train_datagen.flow_from_directory(
        args.data_dir + '/train',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    
    validation_generator = train_datagen.flow_from_directory(
        args.data_dir + '/train',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    
    # Load test data if available
    if os.path.exists(args.data_dir + '/test'):
        test_generator = test_datagen.flow_from_directory(
            args.data_dir + '/test',
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
    else:
        test_generator = None
    
    # Get the number of classes
    num_classes = len(train_generator.class_indices)
    print(f"Number of classes: {num_classes}")
    print(f"Class indices: {train_generator.class_indices}")
    
    # Create model directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Create the model
    model = create_model(input_shape=(img_width, img_height, 3), num_classes=num_classes)
    
    # Set up callbacks
    checkpoint = ModelCheckpoint(
        'models/coffee_disease_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    callbacks = [checkpoint, early_stopping, reduce_lr]
    
    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=args.epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        callbacks=callbacks
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate on test set if available
    if test_generator:
        print("\nEvaluating on test set...")
        test_loss, test_accuracy = model.evaluate(test_generator)
        print(f"Test accuracy: {test_accuracy:.4f}")
        print(f"Test loss: {test_loss:.4f}")
    
    # Fine-tune the model if specified
    if args.fine_tune:
        print("\nFine-tuning the model...")
        # Unfreeze some layers of the base model
        for layer in model.layers[0].layers[-20:]:  # Unfreeze the last 20 layers
            layer.trainable = True
            
        # Recompile the model with a lower learning rate
        model.compile(
            optimizer=Adam(learning_rate=1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train the model again
        fine_tune_history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            epochs=args.fine_tune_epochs,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // batch_size,
            callbacks=callbacks
        )
        
        # Evaluate on test set after fine-tuning
        if test_generator:
            print("\nEvaluating fine-tuned model on test set...")
            test_loss, test_accuracy = model.evaluate(test_generator)
            print(f"Test accuracy after fine-tuning: {test_accuracy:.4f}")
            print(f"Test loss after fine-tuning: {test_loss:.4f}")
    
    # Save the class indices for later use
    import json
    with open('models/class_indices.json', 'w') as f:
        json.dump(train_generator.class_indices, f)
        
    print("Training completed successfully!")

def main():
    parser = argparse.ArgumentParser(description='Train a model for coffee leaf disease detection')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing the dataset')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--fine_tune', action='store_true', help='Whether to fine-tune the model after initial training')
    parser.add_argument('--fine_tune_epochs', type=int, default=10, help='Number of fine-tuning epochs')
    
    args = parser.parse_args()
    train_model(args)

if __name__ == '__main__':
    # Set memory growth for GPUs
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    main()  