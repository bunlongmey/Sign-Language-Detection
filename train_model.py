import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import SGD

# -----------------------------
# Paths & parameters
# -----------------------------
input_dir = './processed_data'  # preprocessed images
batch_size = 32
image_size = (224, 224)

# -----------------------------
# Load dataset
# -----------------------------
train_data = tf.keras.utils.image_dataset_from_directory(
    input_dir,
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=image_size,
    batch_size=batch_size
)

val_data = tf.keras.utils.image_dataset_from_directory(
    input_dir,
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=image_size,
    batch_size=batch_size
)

class_names = train_data.class_names
print("Classes:", class_names)

# -----------------------------
# Normalize the data
# -----------------------------
normalization_layer = tf.keras.layers.Rescaling(1./255)

train_data = train_data.map(lambda x, y: (normalization_layer(x), y))
val_data = val_data.map(lambda x, y: (normalization_layer(x), y))

# -----------------------------
# Load VGG16 base
# -----------------------------
base_model = VGG16(weights='imagenet', include_top=False, input_shape=image_size + (3,))

# Freeze base model layers (optional fine-tuning)
for layer in base_model.layers[:-4]:
    layer.trainable = False
for layer in base_model.layers[-4:]:
    layer.trainable = True

# -----------------------------
# Build model
# -----------------------------
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(class_names), activation='softmax')
])

# -----------------------------
# Compile model
# -----------------------------
model.compile(
    optimizer=SGD(learning_rate=0.001, momentum=0.9),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# Learning rate scheduler
# -----------------------------
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, min_lr=1e-6)

# -----------------------------
# Train model
# -----------------------------
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,  # start small; increase later
    callbacks=[lr_scheduler]
)

# -----------------------------
# Save model
# -----------------------------
model.save('./models/asl_model.h5')
print("Model saved to ./models/asl_model.h5")

# import tensorflow as tf
# from tensorflow.keras import models, layers, callbacks, optimizers, applications

# # -----------------------------
# # Dataset settings
# # -----------------------------
# input_dir = './processed_data'
# batch_size = 32
# image_size = (224, 224)

# # -----------------------------
# # Load datasets
# # -----------------------------
# train_data = tf.keras.utils.image_dataset_from_directory(
#     input_dir,
#     validation_split=0.2,
#     subset="training",
#     seed=123,
#     image_size=image_size,
#     batch_size=batch_size
# )

# val_data = tf.keras.utils.image_dataset_from_directory(
#     input_dir,
#     validation_split=0.2,
#     subset="validation",
#     seed=123,
#     image_size=image_size,
#     batch_size=batch_size
# )

# class_names = train_data.class_names
# print("Classes:", class_names)

# # -----------------------------
# # Normalize data
# # -----------------------------
# normalization_layer = layers.Rescaling(1.0 / 255)

# train_data = train_data.map(lambda x, y: (normalization_layer(x), y))
# val_data = val_data.map(lambda x, y: (normalization_layer(x), y))

# # Improve performance
# AUTOTUNE = tf.data.AUTOTUNE
# train_data = train_data.prefetch(AUTOTUNE)
# val_data = val_data.prefetch(AUTOTUNE)

# # -----------------------------
# # Load VGG16 base model
# # -----------------------------
# base_model = applications.VGG16(
#     weights='imagenet',
#     include_top=False,
#     input_shape=image_size + (3,)
# )

# base_model.trainable = False  # freeze base model

# # -----------------------------
# # Build model
# # -----------------------------
# model = models.Sequential([
#     base_model,
#     layers.GlobalAveragePooling2D(),
#     layers.Dense(128, activation='relu'),
#     layers.Dropout(0.3),
#     layers.Dense(len(class_names), activation='softmax')
# ])

# # -----------------------------
# # Compile model
# # -----------------------------
# model.compile(
#     optimizer=optimizers.SGD(learning_rate=0.001, momentum=0.9),
#     loss='sparse_categorical_crossentropy',
#     metrics=['accuracy']
# )

# # -----------------------------
# # Learning rate scheduler
# # -----------------------------
# lr_scheduler = callbacks.ReduceLROnPlateau(
#     monitor='val_loss',
#     patience=3,
#     factor=0.5,
#     min_lr=1e-6
# )

# # -----------------------------
# # Train model
# # -----------------------------
# history = model.fit(
#     train_data,
#     validation_data=val_data,
#     epochs=50,
#     callbacks=[lr_scheduler]
# )

# # -----------------------------
# # Save model
# # -----------------------------
# model.save('./models/asl_model.h5')
# # import tensorflow as tf
# # from tensorflow.keras.models import Sequential
# # from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
# # from tensorflow.keras.callbacks import ReduceLROnPlateau
# # from tensorflow.keras.applications import VGG16
# # from tensorflow.keras.optimizers import SGD
# # from tensorflow.keras.layers import Rescaling

# # # -----------------------------
# # # Dataset settings
# # # -----------------------------
# # input_dir = './processed_data'
# # batch_size = 32
# # image_size = (224, 224)

# # # -----------------------------
# # # Load datasets
# # # -----------------------------
# # train_data = tf.keras.utils.image_dataset_from_directory(
# #     input_dir,
# #     validation_split=0.2,
# #     subset="training",
# #     seed=123,
# #     image_size=image_size,
# #     batch_size=batch_size
# # )

# # val_data = tf.keras.utils.image_dataset_from_directory(
# #     input_dir,
# #     validation_split=0.2,
# #     subset="validation",
# #     seed=123,
# #     image_size=image_size,
# #     batch_size=batch_size
# # )

# # class_names = train_data.class_names
# # print("Classes:", class_names)

# # # -----------------------------
# # # Normalize data
# # # -----------------------------
# # normalization_layer = Rescaling(1.0 / 255)

# # train_data = train_data.map(lambda x, y: (normalization_layer(x), y))
# # val_data = val_data.map(lambda x, y: (normalization_layer(x), y))

# # # Improve performance
# # AUTOTUNE = tf.data.AUTOTUNE
# # train_data = train_data.prefetch(buffer_size=AUTOTUNE)
# # val_data = val_data.prefetch(buffer_size=AUTOTUNE)

# # # -----------------------------
# # # Load VGG16 base model
# # # -----------------------------
# # base_model = VGG16(
# #     weights='imagenet',
# #     include_top=False,
# #     input_shape=image_size + (3,)
# # )

# # # Freeze base model first
# # base_model.trainable = False

# # # -----------------------------
# # # Build model
# # # -----------------------------
# # model = Sequential([
# #     base_model,
# #     GlobalAveragePooling2D(),
# #     Dense(128, activation='relu'),
# #     Dropout(0.3),
# #     Dense(len(class_names), activation='softmax')
# # ])

# # # -----------------------------
# # # Compile model
# # # -----------------------------
# # model.compile(
# #     optimizer=SGD(learning_rate=0.001, momentum=0.9),
# #     loss='sparse_categorical_crossentropy',
# #     metrics=['accuracy']
# # )

# # # -----------------------------
# # # Learning rate scheduler
# # # -----------------------------
# # lr_scheduler = ReduceLROnPlateau(
# #     monitor='val_loss',
# #     patience=3,
# #     factor=0.5,
# #     min_lr=1e-6
# # )

# # # -----------------------------
# # # Train model
# # # -----------------------------
# # history = model.fit(
# #     train_data,
# #     validation_data=val_data,
# #     epochs=50,
# #     callbacks=[lr_scheduler]
# # )

# # # -----------------------------
# # # Save model
# # # -----------------------------
# # model.save('./models/asl_model.h5')




# # # import tensorflow as tf
# # # from tensorflow.keras.models import Sequential 
# # # from tensorflow.keras.layers import Dropout, GlobalAveragePooling20
# # # from tensorflow.keras.callbacks import ReduceLROnPlateau
# # # from tensorflow.keras.applications import VGG16
# # # from tensorflow.keras.optimizers import SGD

# # # #dataseth path
# # # input_dir = './processed_data'
# # # batch_size = 32
# # # image_size = (224, 224)

# # # #load dataset and retrieve class names
# # # train_data = tf.keras.utils.image_dataset_from_directory(
# # #     input_dir,
# # #     validation_split=0.2
# # #     subset='validation',
# # #     seed=123,
# # #     image_size = image_size,
# # #     batch_size=batch_size 
# # # )

# # # class_names = train_data.class_names

# # # # normalize the data
# # # normalization_layer = tf.keras.layer.Rescaling(1./255)
# # # train_data = train_data.map(lambda x, y: (normalization_lay(x), y))
# # # val_data = val_data.map(lamda x, y: (normaliztion_layer(x), (y)))

# # # # load pre-trained VGG16 model without the top layer
# # # base_model = VGG16(weights='imagenet', include_top=False,input_shape=image_size + (3,))

# # # #unfreeze some layers for fine-tuning
# # # for layer in base_model.layers[-4:]:
# # #     layer.trainable = True

# # # model = Sequential ([
# # #     base_model, 
# # #     GlobalAveragePooling20()
# # #     Dropout(0,3),
# # #     Dense (128, activation='relu'),
# # #     Dropout(0, 3),
# # #     Dense(len(class_names), activation='softmax')
# # # ])

# # # model.cpmpile(optimizer=SGD(learning_rate=0.001, momentum=0.9), loss='spare_categorical_crossentropy', metrics=[''])
# # # lr_scheduler = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, min_lr=1e-6)

# # # history = model.fit(
# # #     train_data,
# # #     validation_data=val_data,
# # #     epochs=50,
# # #     callbacks=[lr_scheduler]
# # # )


# # # model.save('./models/asl_model.h5')