import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
import warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_curve, roc_auc_score, confusion_matrix,
    classification_report, log_loss
)

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Reshape,
    LeakyReLU, GlobalAveragePooling2D, GlobalAveragePooling1D, Multiply,
    Layer, MultiHeadAttention, LayerNormalization, Add, Concatenate,
    BatchNormalization
)
from tensorflow.keras.models import Model

warnings.filterwarnings('ignore')

# https://github.com/kobiso/CBAM-keras
# https://github.com/titu1994/keras-squeeze-excite-network

class DenseBlock(Layer):
    def __init__(self, num_layers, growth_rate, kernel_size=3):
        super(DenseBlock, self).__init__()
        self.num_layers = num_layers
        self.growth_rate = growth_rate
        self.kernel_size = kernel_size
        self.conv_layers = [
            Conv2D(growth_rate, kernel_size=kernel_size, padding='same') 
            for _ in range(num_layers)
        ]

    def call(self, inputs):
        x = inputs
        outputs = [x]
        for conv in self.conv_layers:
            x = conv(x)
            x = LeakyReLU(alpha=0.05)(x)
            outputs.append(x)
            x = Concatenate(axis=-1)(outputs)
        return x


class AttentionModule(Layer):
    def __init__(self, num_heads):
        super(AttentionModule, self).__init__()
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=256)
        self.layer_norm = LayerNormalization()

    def call(self, patches):
        attn_output = self.attention(patches, patches)
        return self.layer_norm(attn_output + patches)

class DenseBlock(Layer):
    def __init__(self, num_layers=3, growth_rate=32, **kwargs):
        super().__init__(**kwargs)
        self.blocks = []
        for _ in range(num_layers):
            self.blocks.append(
                tf.keras.Sequential([
                    Conv2D(growth_rate, 3, padding='same', use_bias=False),
                    BatchNormalization(),
                    LeakyReLU(0.05)
                ])
            )

    def call(self, x):
        feats = [x]
        for block in self.blocks:
            y = block(x)
            feats.append(y)
            x = Concatenate()(feats)
        return x


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
        x = self.fuse(x)
        return x


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
        x = self.norm2(x + f)
        return x


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


def load_and_preprocess_images(directory, label, img_rows=512, img_cols=512):
    images = []
    labels = []
    files = os.listdir(directory)
    for f in files:
        img = cv2.imread(os.path.join(directory, f), 1)
        if img is not None:
            img = cv2.resize(img, (img_rows, img_cols))
            images.append(img)
            labels.append(label)
    return images, labels



X_data, Y_data = [], []

data_dirs = {
    '/kaggle/input/kneemeddataset/KneeMedDataset/deepfake': 0,
    '/kaggle/input/kneemeddataset/KneeMedDataset/real': 1
}

for directory, label in data_dirs.items():
    images, labels = load_and_preprocess_images(directory, label, img_rows=224, img_cols=224)
    X_data.extend(images)
    Y_data.extend(labels)

X_data = np.array(X_data)
Y_data = np.array(Y_data)

img_rows, img_cols = 224, 224
if tf.keras.backend.image_data_format() == 'channels_first':
    X_data = X_data.reshape(X_data.shape[0], 3, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    X_data = X_data.reshape(X_data.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)

X_data = X_data.astype('float32') / 255.0


X_train_full, X_test, Y_train_full, Y_test = train_test_split(
    X_data, Y_data, test_size=0.2, random_state=4, stratify=Y_data
)

X_train, X_val, Y_train, Y_val = train_test_split(
    X_train_full, Y_train_full, test_size=0.1, random_state=42, stratify=Y_train_full
)

print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")
print(f"Testing data shape: {X_test.shape}")

def get_hybrid_model_dense(input_shape):
    inputs = Input(shape=input_shape)

 
    cnn = Conv2D(32, 3, padding='same', activation='relu')(inputs)
    cnn = DenseBlock(3, 32)(cnn)
    cnn = SEBlock()(cnn)
    cnn = MaxPooling2D(2)(cnn)

    cnn = DenseBlock(3, 64)(cnn)
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
    model = Model(inputs, outputs, name="Hybrid_Model_Dense")
    return model


def compute_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer = fnr[np.nanargmin(np.abs(fnr - fpr))]
    return eer



early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)
csv_logger = CSVLogger('training_history.csv', append=False)


model = get_hybrid_model_dense(input_shape)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train, Y_train,
    batch_size=16,
    epochs=30,
    validation_data=(X_val, Y_val),
    shuffle=True,
    callbacks=[early_stopping, csv_logger]
)


y_pred_proba = model.predict(X_test).flatten()
y_pred = (y_pred_proba > 0.5).astype(int)

accuracy = accuracy_score(Y_test, y_pred)
precision = precision_score(Y_test, y_pred)
recall = recall_score(Y_test, y_pred)
f1 = f1_score(Y_test, y_pred)
eer = compute_eer(Y_test, y_pred_proba)
mcc = matthews_corrcoef(Y_test, y_pred)

print("\nEvaluation Metrics:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"EER:       {eer:.4f}")
print(f"MCC:       {mcc:.4f}\n")

class_report = classification_report(Y_test, y_pred, target_names=['Deepfake', 'Real'])
print("Classification Report:\n", class_report)

# Confusion Matrix Plot
cm = confusion_matrix(Y_test, y_pred)
cm_sum = np.sum(cm, axis=1, keepdims=True)
cm_perc = cm / cm_sum.astype(float) * 100

annot = np.empty_like(cm).astype(str)
nrows, ncols = cm.shape
for i in range(nrows):
    for j in range(ncols):
        c = cm[i, j]
        p = cm_perc[i, j]
        annot[i, j] = f'{c}\n({p:.1f}%)'

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=annot, fmt='', cmap='BuPu',
    xticklabels=['Deepfake', 'Real'],
    yticklabels=['Deepfake', 'Real'],
    linewidths=1.5, linecolor='black', cbar=True,
    annot_kws={"size": 13, "weight": "bold"}
)
plt.xlabel('Predicted Class', fontsize=12, weight='bold')
plt.ylabel('Actual Class', fontsize=12, weight='bold')
plt.title('Confusion Matrix', fontsize=14, weight='bold')
plt.savefig("Confusion matrix.png")
plt.tight_layout()
plt.show()


fpr, tpr, thresholds = roc_curve(Y_test, y_pred_proba)
roc_auc = roc_auc_score(Y_test, y_pred_proba)

fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
ax.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.5f)' % roc_auc)
ax.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
ax.set_xlim([-0.01, 1.01])
ax.set_ylim([0.0, 1.05])
ax.set_xticks(np.arange(0.0, 1.1, 0.1))
ax.set_yticks(np.arange(0.0, 1.1, 0.1))
ax.set_xlabel('False Positive Rate (FPR)', fontsize=12)
ax.set_ylabel('True Positive Rate (TPR)', fontsize=12)
ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
ax.legend(loc="lower right")
ax.grid(True)
plt.savefig("ROC.png")
plt.show()

# Training Curves Plot
plt.style.use('bmh')
plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue', linestyle='-', marker='o', markersize=5)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange', linestyle='--', marker='x', markersize=5)
plt.title('Model Accuracy', fontsize=18)
plt.xlabel('Epochs', fontsize=18)
plt.ylabel('Accuracy', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.ylim(0, 1.1)
plt.grid(linewidth=2)
plt.legend(fontsize=18)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', color='red', linestyle='-', marker='o', markersize=5)
plt.plot(history.history['val_loss'], label='Validation Loss', color='green', linestyle='--', marker='x', markersize=5)
plt.title('Model Loss', fontsize=18)
plt.xlabel('Epochs', fontsize=18)
plt.ylabel('Loss', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid(linewidth=2)
plt.legend(fontsize=18)

plt.savefig('training_validation_curves1.png', dpi=1000, bbox_inches='tight')
plt.show()
