import os
import random
import warnings
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(SEED)
np.random.seed(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)

from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Reshape,
    LeakyReLU, GlobalAveragePooling2D, GlobalAveragePooling1D, Multiply,
    Layer, MultiHeadAttention, LayerNormalization, BatchNormalization, Add, Concatenate
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_curve, roc_auc_score, confusion_matrix, classification_report
)

warnings.filterwarnings('ignore')



def load_and_preprocess_images(directory, label, img_rows=224, img_cols=224):
    images, labels = [], []
    if not os.path.exists(directory):
        print(f"Warning: Directory not found: {directory}")
        return images, labels
        
    files = os.listdir(directory)
    for f in files:
        file_path = os.path.join(directory, f)
        img = cv2.imread(file_path, 1)
        if img is not None:
            img = cv2.resize(img, (img_rows, img_cols))
            images.append(img)
            labels.append(label)
    return images, labels

train_dirs = {
    '/content/DatasetLungs/Train/Real': 0,
    '/content/DatasetLungs/Train/Deepfake': 1,
}

test_dirs = {
    '/content/DatasetLungs/Test/Real': 0,
    '/content/DatasetLungs/Test/Deepfake': 1,
}

X_train, Y_train = [], []
X_test, Y_test = [], []

for directory, label in train_dirs.items():
    images, labels = load_and_preprocess_images(directory, label, img_rows=224, img_cols=224)
    X_train.extend(images)
    Y_train.extend(labels)

for directory, label in test_dirs.items():
    images, labels = load_and_preprocess_images(directory, label, img_rows=224, img_cols=224)
    X_test.extend(images)
    Y_test.extend(labels)

X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)

img_rows, img_cols = 224, 224
if tf.keras.backend.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 3, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 3, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Split validation set from training data
X_train, X_valid, Y_train, Y_valid = train_test_split(
    X_train,
    Y_train,
    test_size=0.1,
    random_state=SEED,
    stratify=Y_train
)

class ResNetBlock(Layer):
    def __init__(self, filters, stride=1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.stride = stride
        self.conv1 = Conv2D(filters, 3, strides=stride, padding='same', use_bias=False)
        self.bn1 = LayerNormalization()
        self.act1 = LeakyReLU(0.05)
        self.conv2 = Conv2D(filters, 3, padding='same', use_bias=False)
        self.bn2 = LayerNormalization()

    def build(self, input_shape):
        in_channels = input_shape[-1]
        if in_channels != self.filters or self.stride != 1:
            self.shortcut = tf.keras.Sequential([
                Conv2D(self.filters, 1, strides=self.stride, padding='same', use_bias=False),
                BatchNormalization()
            ])
        else:
            self.shortcut = lambda x: x
        super().build(input_shape)

    def call(self, x):
        shortcut = self.shortcut(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = Add()([x, shortcut])
        return LeakyReLU(0.05)(x)

class SEBlock(Layer):
    def __init__(self, ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        c = input_shape[-1]
        self.gap = GlobalAveragePooling2D()
        self.fc1 = Dense(max(c // self.ratio, 8), activation='relu')
        self.fc2 = Dense(c, activation='sigmoid')
        self.reshape = Reshape((1, 1, c))
        super().build(input_shape)

    def call(self, x):
        s = self.gap(x)
        s = self.fc1(s)
        s = self.fc2(s)
        s = self.reshape(s)
        return Multiply()([x, s])

class MultiScalePatchEmbedding(Layer):
    def __init__(self, embed_dim=128, patch_sizes=[8, 16, 32], **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.patch_sizes = patch_sizes
        self.proj_layers = [
            Conv2D(embed_dim, kernel_size=p, strides=p, padding='valid')
            for p in patch_sizes
        ]
        self.fuse = Dense(embed_dim)

    def call(self, x):
        multi_feats = []
        for conv in self.proj_layers:
            f = conv(x)
            h = tf.shape(f)[1]
            w = tf.shape(f)[2]
            f = tf.reshape(f, (-1, h * w, tf.shape(f)[-1]))
            multi_feats.append(f)
        x = Concatenate(axis=1)(multi_feats)
        return self.fuse(x)

class TransformerBlock(Layer):
    def __init__(self, embed_dim=128, num_heads=4, **kwargs):
        super().__init__(**kwargs)
        self.attn = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()
        self.ffn = tf.keras.Sequential([
            Dense(embed_dim * 2, activation='gelu'),
            Dense(embed_dim)
        ])

    def call(self, x):
        a = self.attn(x, x)
        x = self.norm1(x + a)
        f = self.ffn(x)
        return self.norm2(x + f)

class CrossAttentionFusion(Layer):
    def __init__(self, embed_dim=128, num_heads=4, **kwargs):
        super().__init__(**kwargs)
        self.attn = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.norm = LayerNormalization()

    def call(self, cnn_feat, vit_feat):
        cnn_feat = tf.expand_dims(cnn_feat, axis=1)
        vit_feat = tf.expand_dims(vit_feat, axis=1)
        attn_out = self.attn(query=cnn_feat, key=vit_feat, value=vit_feat)
        fused = self.norm(attn_out + cnn_feat)
        return tf.squeeze(fused, axis=1)



def get_hybrid_modelResNet(input_shape):
    inputs = Input(shape=input_shape)

    cnn = Conv2D(32, 3, padding='same', activation='relu')(inputs)
    cnn = ResNetBlock(32)(cnn)
    cnn = ResNetBlock(32)(cnn)
    cnn = SEBlock()(cnn)
    cnn = MaxPooling2D(2)(cnn)

    cnn = ResNetBlock(64)(cnn)
    cnn = ResNetBlock(64)(cnn)
    cnn = SEBlock()(cnn)
    cnn = MaxPooling2D(2)(cnn)

    cnn = Conv2D(128, 3, padding='same', activation='relu')(cnn)
    cnn = GlobalAveragePooling2D()(cnn)


    vit = MultiScalePatchEmbedding(embed_dim=128, patch_sizes=[8, 16, 32])(inputs)
    vit = TransformerBlock(embed_dim=128, num_heads=4)(vit)
    vit = TransformerBlock(embed_dim=128, num_heads=4)(vit)
    vit = GlobalAveragePooling1D()(vit)


    fusion = CrossAttentionFusion(embed_dim=128, num_heads=4)(cnn, vit)
    fusion = Concatenate()([fusion, cnn, vit])

   
    fusion = Dense(256, activation='relu')(fusion)
    fusion = Dropout(0.3)(fusion)
    fusion = Dense(128, activation='relu')(fusion)
    fusion = Dropout(0.2)(fusion)

    outputs = Dense(1, activation='sigmoid')(fusion)

    return Model(inputs, outputs, name="Hybrid_ResNet_ViT_CrossAttention")

model = get_hybrid_modelResNet(input_shape)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
]

history = model.fit(
    X_train, Y_train,
    batch_size=8,
    epochs=50,
    validation_data=(X_valid, Y_valid),
    callbacks=callbacks
)



def compute_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_index = np.nanargmin(np.abs(fnr - fpr))
    return fnr[eer_index]

y_pred_proba = model.predict(X_test).flatten()
y_pred = (y_pred_proba > 0.5).astype(int)

accuracy = accuracy_score(Y_test, y_pred)
precision = precision_score(Y_test, y_pred)
recall = recall_score(Y_test, y_pred)
f1 = f1_score(Y_test, y_pred)
eer = compute_eer(Y_test, y_pred_proba)
mcc = matthews_corrcoef(Y_test, y_pred)

print("\n" + "="*40)
print("TEST EVALUATION RESULTS")
print("="*40)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"EER:       {eer:.4f}")
print(f"MCC:       {mcc:.4f}")

print("\nClassification Report:\n")
print(classification_report(Y_test, y_pred, target_names=['Real', 'Deepfake']))
