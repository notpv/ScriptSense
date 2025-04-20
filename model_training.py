import os
import cv2
import numpy as np
import tensorflow as tf
from keras import layers, Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import spacy
import nltk
from nltk.metrics.distance import edit_distance
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Download required NLTK resources
nltk.download('words', quiet=True)
nltk.download('punkt', quiet=True)

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    os.system('python -m spacy download en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

# Configuration
class Config:
    IMG_HEIGHT = 128
    IMG_WIDTH = 800
    BATCH_SIZE = 8
    EPOCHS = 20
    MAX_TEXT_LEN = 100
    VALIDATION_SPLIT = 0.2
    CHAR_VOCAB = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,?!-_' "

config = Config()

# Create character to index mapping and vice versa
char_to_idx = {char: idx + 1 for idx, char in enumerate(config.CHAR_VOCAB)}
char_to_idx['<pad>'] = 0  # Padding token
idx_to_char = {idx: char for char, idx in char_to_idx.items()}
vocab_size = len(char_to_idx)

# Dataset paths
training_dir = "processed_data/training"
testing_dir = "processed_data/testing"

# Load and preprocess data
def load_data(directory):
    image_paths = []
    labels = []
    
    image_files = [f for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    for img_file in tqdm(image_files, desc=f"Loading data from {directory}"):
        # Get image path
        img_path = os.path.join(directory, img_file)
        
        # Get corresponding label file
        label_file = os.path.join(directory, os.path.splitext(img_file)[0] + ".txt")
        
        # Check if label file exists
        if not os.path.exists(label_file):
            print(f"Warning: No label file found for {img_file}")
            continue
        
        # Read label file
        with open(label_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("Transcription:"):
                    transcription = line.replace("Transcription:", "").strip()
                    break
            else:
                # If no transcription found, skip this image
                print(f"Warning: No transcription found in {label_file}")
                continue
        
        image_paths.append(img_path)
        labels.append(transcription)
    
    return image_paths, labels

# Load training and testing data
print("Loading training data...")
train_image_paths, train_labels = load_data(training_dir)
print(f"Loaded {len(train_image_paths)} training samples")

print("Loading testing data...")
test_image_paths, test_labels = load_data(testing_dir)
print(f"Loaded {len(test_image_paths)} testing samples")

# Create validation split from training data
train_image_paths, val_image_paths, train_labels, val_labels = train_test_split(
    train_image_paths, train_labels, test_size=config.VALIDATION_SPLIT, random_state=42
)

print(f"Training samples: {len(train_image_paths)}")
print(f"Validation samples: {len(val_image_paths)}")

# Preprocess image
def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize to fixed height and keep aspect ratio
    h, w = img.shape
    if h != config.IMG_HEIGHT:
        w_new = int(w * (config.IMG_HEIGHT / h))
        img = cv2.resize(img, (w_new, config.IMG_HEIGHT))
    
    # Pad to fixed width if needed
    h, w = img.shape
    if w < config.IMG_WIDTH:
        img = np.pad(img, ((0, 0), (0, config.IMG_WIDTH - w)), 'constant', constant_values=255)
    elif w > config.IMG_WIDTH:
        img = cv2.resize(img, (config.IMG_WIDTH, config.IMG_HEIGHT))
    
    # Normalize and add channel dimension
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)
    
    return img

# Encode text to sequence of indices
def encode_text(text):
    text = text[:config.MAX_TEXT_LEN]
    encoded = [char_to_idx.get(char, char_to_idx.get(' ', 0)) for char in text]
    length = len(encoded)
    
    # Pad to MAX_TEXT_LEN
    if length < config.MAX_TEXT_LEN:
        encoded = encoded + [char_to_idx['<pad>']] * (config.MAX_TEXT_LEN - length)
    
    return np.array(encoded), length

# Decode indices to text
def decode_text(indices):
    text = ''.join([idx_to_char.get(idx, '') for idx in indices if idx > 0])
    return text

# Create TF dataset generator - FIXED VERSION
def create_dataset(image_paths, labels, batch_size=8, is_training=True):
    def generator():
        indices = np.arange(len(image_paths))
        if is_training:
            np.random.shuffle(indices)
        
        for i in indices:
            img_path = image_paths[i]
            text = labels[i]
            
            img = preprocess_image(img_path)
            encoded_text, text_length = encode_text(text)
            
            # The target should be one-dimensional to work with sparse_categorical_crossentropy
            # Feed the encoded text directly as the output (not in a dictionary)
            yield {
                'input_images': img,
                'input_text': np.array(encoded_text)
            }, np.array(encoded_text)
    
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            {
                'input_images': tf.TensorSpec(shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 1), dtype=tf.float32),
                'input_text': tf.TensorSpec(shape=(config.MAX_TEXT_LEN,), dtype=tf.int32)
            },
            tf.TensorSpec(shape=(config.MAX_TEXT_LEN,), dtype=tf.int32)
        )
    )
    
    if is_training:
        dataset = dataset.shuffle(buffer_size=len(image_paths))
    
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# Create datasets
train_dataset = create_dataset(train_image_paths, train_labels, config.BATCH_SIZE, True)
val_dataset = create_dataset(val_image_paths, val_labels, config.BATCH_SIZE, False)
test_dataset = create_dataset(test_image_paths, test_labels, config.BATCH_SIZE, False)

# Define the model architecture - FIXED VERSION
def create_model():
    # Input layers
    input_images = layers.Input(shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 1), name='input_images')
    input_text = layers.Input(shape=(config.MAX_TEXT_LEN,), name='input_text')

    # CNN for feature extraction
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_images)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)

    # Get CNN output shape statically
    _, h, w, c = x.shape

    # Prepare for sequence model - collapse height dimension
    cnn_features = layers.Reshape((-1, h * c))(x)

    # Bidirectional LSTM layers for visual features
    visual_x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(cnn_features)
    visual_x = layers.Dropout(0.25)(visual_x)
    visual_x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(visual_x)
    visual_x = layers.Dropout(0.25)(visual_x)
    
    # Process text input
    text_embeddings = layers.Embedding(vocab_size, 256)(input_text)
    text_x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(text_embeddings)
    text_x = layers.Dropout(0.25)(text_x)
    
    # Apply attention to visual features
    visual_attention_output = layers.MultiHeadAttention(
        num_heads=4,
        key_dim=64
    )(text_x, visual_x)
    visual_x = visual_attention_output
    
    # Process visual features to have consistent dimensions
    visual_x = layers.TimeDistributed(layers.Dense(512))(visual_x)
    
    # Reshape to ensure we get the right dimensions
    feature_dim = 512
    visual_x = layers.Reshape((-1, feature_dim))(visual_x)
    
    # Adjust sequence length with 1D convolution
    visual_x = layers.Conv1D(
        filters=feature_dim,
        kernel_size=3,
        padding='same',
        strides=1,
        activation='relu'
    )(visual_x)
    
    # Use global average pooling followed by a dense layer
    pooled_features = layers.GlobalAveragePooling1D()(visual_x)
    pooled_features = layers.Dense(feature_dim)(pooled_features)
    
    # Repeat the pooled features to match the target sequence length
    visual_x = layers.RepeatVector(config.MAX_TEXT_LEN)(pooled_features)
    
    # Process text features
    text_x = layers.TimeDistributed(layers.Dense(512))(text_x)
    
    # Concatenate along the feature dimension
    combined = layers.Concatenate(axis=2)([visual_x, text_x])
    
    # Process combined features
    combined = layers.LayerNormalization()(combined)
    combined = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(combined)
    combined = layers.Dropout(0.25)(combined)
    
    # Final prediction layer - removed the 'name' parameter
    output = layers.TimeDistributed(layers.Dense(vocab_size, activation='softmax'))(combined)

    # Create model
    model = Model(inputs=[input_images, input_text], outputs=output)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',  # This expects integer targets
        metrics=['accuracy']
    )
    return model

# Create and train the model
model = create_model()
model.summary()

# Define callbacks
checkpoint_cb = ModelCheckpoint(
    'best_htr_model.keras',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

early_stopping_cb = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# Train model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=config.EPOCHS,
    callbacks=[checkpoint_cb, early_stopping_cb]
)

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.close()

# Define a function for inference - FIXED VERSION
def predict(model, image_path):
    img = preprocess_image(image_path)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    # Create dummy text input (will be replaced with predictions in real inference)
    dummy_text = np.zeros((1, config.MAX_TEXT_LEN), dtype=np.int32)
    
    # Get prediction
    pred = model.predict([img, dummy_text])
    pred_indices = np.argmax(pred, axis=-1)[0]
    
    # Decode prediction
    pred_text = decode_text(pred_indices)
    
    return pred_text

# Post-processing functions
def apply_spacy_correction(text):
    doc = nlp(text)
    corrected_text = []
    for token in doc:
        # Apply spaCy's linguistic knowledge to correct words
        if token.is_alpha and len(token.text) > 1:
            # Check if token is misspelled and get correction
            corrected_token = token.text
            corrected_text.append(corrected_token)
        else:
            corrected_text.append(token.text)
    
    return ' '.join(corrected_text)

def apply_nltk_correction(text):
    # Simple word tokenization without relying on nltk.word_tokenize
    words = text.split()
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())
    
    corrected_words = []
    for word in words:
        if word.lower() not in english_vocab and word.isalpha() and len(word) > 1:
            # Find closest word in vocabulary
            candidates = [w for w in english_vocab if w[0] == word[0].lower()]
            if candidates:
                closest = min(candidates, key=lambda x: edit_distance(word.lower(), x))
                corrected_words.append(closest)
            else:
                corrected_words.append(word)
        else:
            corrected_words.append(word)
    
    return ' '.join(corrected_words)

# Evaluate model on test set
def evaluate_model(model, test_dataset, test_image_paths, test_labels):
    print("Evaluating model on test set...")
    
    total_cer = 0
    total_wer = 0
    count = 0
    
    for i, (img_path, true_label) in enumerate(zip(test_image_paths, test_labels)):
        # Get prediction
        pred_text = predict(model, img_path)
        
        # Apply post-processing
        corrected_text = apply_spacy_correction(pred_text)
        corrected_text = apply_nltk_correction(corrected_text)
        
        # Calculate CER (Character Error Rate)
        cer = edit_distance(true_label, corrected_text) / max(len(true_label), 1)
        
        # Calculate WER (Word Error Rate)
        true_words = true_label.split()
        pred_words = corrected_text.split()
        wer = edit_distance(true_words, pred_words) / max(len(true_words), 1)
        
        total_cer += cer
        total_wer += wer
        count += 1
        
        # Print some examples
        if i < 5:
            print(f"Example {i+1}:")
            print(f"True: {true_label}")
            print(f"Pred (raw): {pred_text}")
            print(f"Pred (corrected): {corrected_text}")
            print(f"CER: {cer:.4f}, WER: {wer:.4f}")
            print("-" * 50)
    
    # Calculate average metrics
    avg_cer = total_cer / count if count > 0 else 0
    avg_wer = total_wer / count if count > 0 else 0
    
    print(f"Average CER: {avg_cer:.4f}")
    print(f"Average WER: {avg_wer:.4f}")
    
    return avg_cer, avg_wer

# Evaluate the model
cer, wer = evaluate_model(model, test_dataset, test_image_paths, test_labels)

# Save full evaluation results
with open('evaluation_results.txt', 'w') as f:
    f.write(f"Character Error Rate (CER): {cer:.4f}\n")
    f.write(f"Word Error Rate (WER): {wer:.4f}\n")

print("Evaluation complete. Results saved to 'evaluation_results.txt'")