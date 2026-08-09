import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef,
    roc_curve, roc_auc_score, confusion_matrix, classification_report
)

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Reshape,
    LeakyReLU, GlobalAveragePooling2D, Multiply, Layer, MultiHeadAttention,
    LayerNormalization, BatchNormalization, Add, Concatenate,SpatialDropout2D
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import CSVLogger
from sklearn.metrics import confusion_matrix
import seaborn as sns
from keras.utils import to_categorical
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, auc
from itertools import cycle

warnings.filterwarnings('ignore')


#https://github.com/kobiso/CBAM-keras
#https://github.com/titu1994/keras-squeeze-excite-network


from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Conv2D

def ResidualBlock(filters):
    def block(x_input):
        x = Conv2D(filters, (3, 3), padding='same', kernel_regularizer=l2(1e-4))(x_input)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.05)(x)
        x = SpatialDropout2D(0.2)(x)

        x = Conv2D(filters, (3, 3), padding='same', kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization()(x)

        # If number of filters doesn't match, project input
        if x_input.shape[-1] != filters:
            x_input = Conv2D(filters, (1,1), padding='same', kernel_regularizer=l2(1e-4))(x_input)
            x_input = BatchNormalization()(x_input)

        x = Add()([x, x_input])
        x = LeakyReLU(alpha=0.05)(x)
        return x
    return block


class AttentionModule(Layer):
    def __init__(self, num_heads):
        super(AttentionModule, self).__init__()
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=256)
        self.layer_norm = LayerNormalization()

    def call(self, patches):
        attn_output = self.attention(patches, patches)
        return self.layer_norm(attn_output + patches)  
        

class SEBlock(Layer):
    def __init__(self, ratio=16):
        super(SEBlock, self).__init__()
        self.ratio = ratio
        self.global_avg_pool = GlobalAveragePooling2D()
        self.fc1 = None  # Will initialize later
        self.fc2 = None

    def build(self, input_shape):
        channels = input_shape[-1]
        self.fc1 = Dense(channels // self.ratio, activation='relu')
        self.fc2 = Dense(channels, activation='sigmoid')
        self.reshape = Reshape((1, 1, channels))

    def call(self, inputs):
        se = self.global_avg_pool(inputs)
        se = self.fc1(se)
        se = self.fc2(se)
        se = self.reshape(se)
        return Multiply()([inputs, se])
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



X_train, Y_train = [], []
X_test, Y_test = [], []
X_data, Y_data = [], []
X_train, Y_train = [], []
X_test, Y_test = [], []

num_classes = 3  
data_dirs = {
    '/kaggle/input/meddata/MultiClass/MultiClass/Real': 0,
    '/kaggle/input/meddata/MultiClass/MultiClass/FM': 1,
    '/kaggle/input/meddata/MultiClass/MultiClass/FB': 2,  
   
}

for directory, label in data_dirs.items():
    images, labels = load_and_preprocess_images(directory, label, img_rows=224, img_cols=224)
    X_data.extend(images)
    Y_data.extend(labels)


X_data = np.array(X_data)
Y_data = np.array(Y_data)


img_rows, img_cols = 224,224
if tf.keras.backend.image_data_format() == 'channels_first':
    X_data = X_data.reshape(X_data.shape[0], 3, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    X_data = X_data.reshape(X_data.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)

X_data = X_data.astype('float32') / 255.0
Y_data = to_categorical(Y_data, num_classes=num_classes)
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=4,stratify=Y_data)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")


class DenseBlock(Layer):

    def __init__(self,
                 num_layers=3,
                 growth_rate=32):

        super().__init__()

        self.blocks = []

        for _ in range(num_layers):

            self.blocks.append(
                tf.keras.Sequential([
                    Conv2D(growth_rate,
                           3,
                           padding='same',
                           use_bias=False),

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

    def __init__(self, ratio=16):
        super().__init__()
        self.ratio = ratio

    def build(self, input_shape):

        c = input_shape[-1]

        self.gap = GlobalAveragePooling2D()

        self.fc1 = Dense(max(c // self.ratio, 8),
                         activation='relu')

        self.fc2 = Dense(c,
                         activation='sigmoid')

        self.reshape = Reshape((1,1,c))

    def call(self, x):

        s = self.gap(x)
        s = self.fc1(s)
        s = self.fc2(s)
        s = self.reshape(s)

        return Multiply()([x,s])



class MultiScalePatchEmbedding(Layer):
    def __init__(self, embed_dim=128, patch_sizes=[8, 16, 32]):
        super().__init__()

        self.embed_dim = embed_dim
        self.patch_sizes = patch_sizes

        # split embedding properly but ensure recombination = 128
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
            f = tf.reshape(f, (-1, h * w, f.shape[-1]))
            multi_feats.append(f)

        # concatenate tokens
        x = Concatenate(axis=1)(multi_feats)

        # project back to correct dimension (128)
        x = self.fuse(x)

        return x

class TransformerBlock(Layer):

    def __init__(self,
                 embed_dim=128,
                 num_heads=4):

        super().__init__()

        self.attn = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim
        )

        self.norm1 = LayerNormalization()

        self.norm2 = LayerNormalization()

        self.ffn = tf.keras.Sequential([
            Dense(embed_dim*2,
                  activation='gelu'),
            Dense(embed_dim)
        ])

    def call(self, x):

        a = self.attn(x,x)

        x = self.norm1(x+a)

        f = self.ffn(x)

        x = self.norm2(x+f)

        return x

 
class CrossAttentionFusion(Layer):

    def __init__(self,
                 embed_dim=128,
                 num_heads=4):
        super().__init__()

        self.attn = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim
        )

        self.norm = LayerNormalization()

    def call(self, cnn_feat, vit_feat):

        cnn_feat = tf.expand_dims(cnn_feat, axis=1)
        vit_feat = tf.expand_dims(vit_feat, axis=1)

        attn_out = self.attn(
            query=cnn_feat,
            key=vit_feat,
            value=vit_feat
        )

        fused = self.norm(attn_out + cnn_feat)

        return tf.squeeze(fused, axis=1)


def get_hybrid_modelResNet(input_shape):

    inputs = Input(shape=input_shape)


    cnn = Conv2D(32, 3, padding='same', activation='relu')(inputs)

    # --- ResNet stages ---
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

 

    vit = MultiScalePatchEmbedding(
        embed_dim=128,
        patch_sizes=[8,16,32]
    )(inputs)

    vit = TransformerBlock(
        embed_dim=128,
        num_heads=4
    )(vit)

    vit = TransformerBlock(
        embed_dim=128,
        num_heads=4
    )(vit)

    vit = GlobalAveragePooling1D()(vit)



 
    fusion = CrossAttentionFusion(
    embed_dim=128,
    num_heads=4
)(cnn, vit)

    fusion = Concatenate()([
    fusion,
    cnn,
    vit])
    fusion = Dense(256, activation='relu')(fusion)
    fusion = Dropout(0.3)(fusion)
    fusion = Dense(128, activation='relu')(fusion)
    fusion = Dropout(0.2)(fusion)

    outputs = Dense(3, activation='softmax')(fusion)

    model = Model(inputs, outputs)

    return model 

def compute_multiclass_eer(y_true, y_pred_proba, num_classes):
    class_eers = []
    for i in range(num_classes):
        y_true_binary = (y_true == i).astype(int)
        y_pred_proba_class = y_pred_proba[:, i] 
        fpr, tpr, thresholds = roc_curve(y_true_binary, y_pred_proba_class)
        fnr = 1 - tpr
        eer_threshold = thresholds[np.nanargmin(np.abs(fnr - fpr))]
        eer = fnr[np.nanargmin(np.abs(fnr - fpr))]
        class_eers.append(eer)
    return np.mean(class_eers) 


num_folds = 3
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
metrics = {
    'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [],
    'eer': [], 'mcc': []
}

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_curve
)
for fold, (train_index, test_index) in enumerate(kf.split(X_data)):
    print(f"\nTraining fold {fold + 1}/{num_folds}...")


    X_train_full, X_test = X_data[train_index], X_data[test_index]
    Y_train_full, Y_test = Y_data[train_index], Y_data[test_index]


    X_train, X_val, Y_train, Y_val = train_test_split(X_train_full, Y_train_full, test_size=0.1, random_state=42)

  
    model =get_hybrid_modelResNet(input_shape)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

 
    history=model.fit(
        X_train, Y_train,
        batch_size=8,
        epochs=30,
        validation_data=(X_val, Y_val), 
        shuffle=True
    )

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    Y_test_classes = np.argmax(Y_test, axis=1)
    accuracy = accuracy_score(Y_test_classes , y_pred_classes)
    precision = precision_score(Y_test_classes , y_pred_classes,average='macro' )
    recall = recall_score(Y_test_classes , y_pred_classes,average='macro' )
    f1 = f1_score(Y_test_classes , y_pred_classes,average='macro' )

    eer = compute_multiclass_eer(Y_test_classes, y_pred, num_classes)
    mcc = matthews_corrcoef(Y_test_classes, y_pred_classes)

 
    metrics['accuracy'].append(accuracy)
    metrics['precision'].append(precision)
    metrics['recall'].append(recall)
    metrics['f1_score'].append(f1)
    metrics['eer'].append(eer)
    metrics['mcc'].append(mcc)

    print(f"Fold {fold + 1} Metrics:")
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, EER: {eer:.4f}, MCC: {mcc:.4f}")
    results = model.predict(X_test)
    predicted_probabilities = results.flatten()  
    from sklearn.metrics import (
    roc_curve,
    roc_auc_score, confusion_matrix,
    classification_report,
    accuracy_score,
    log_loss,
    matthews_corrcoef)
    target_names = ['Real', 'FM', 'FB']  
    class_report = classification_report(Y_test_classes, y_pred_classes, target_names=target_names)
    print("Classification Report:\n", class_report)


print("\nOverall Metrics (Mean ± Std):")
for metric, values in metrics.items():
    mean = np.mean(values)
    std_dev = np.std(values)
    print(f"{metric.capitalize()}: {mean:.4f} ± {std_dev:.4f}")



cm = confusion_matrix(Y_test_classes, y_pred_classes)
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
sns.heatmap(cm, annot=annot, fmt='', cmap='BuPu',
            xticklabels=target_names,
            yticklabels=target_names,
            linewidths=1.5, linecolor='black', cbar=True,
            annot_kws={"size": 13, "weight": "bold"})

plt.xlabel('Predicted Class', fontsize=12, weight='bold')
plt.ylabel('Actual Class', fontsize=12, weight='bold')
plt.title('Confusion Matrix', fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig("Confusion matrix.png")
plt.show()





Y_test_bin = label_binarize(Y_test_classes, classes=range(num_classes))
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(Y_test_bin[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

colors = cycle(['blue', 'red', 'green', 'purple', 'orange'])
plt.figure(figsize=(7, 6))
for i, color in zip(range(num_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
                   ''.format(target_names[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([-0.01, 1.01])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)', fontsize=12)
plt.ylabel('True Positive Rate (TPR)', fontsize=12)
plt.title('Multi-class ROC Curve', fontsize=14)
plt.legend(loc="lower right", fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.savefig("ROC.png")
plt.show()


import matplotlib.pyplot as plt  
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
