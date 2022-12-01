from model import create_model
from train_val_split import get_train_val_generator

model = create_model()

train_gen, val_gen = get_train_val_generator()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
print(model.summary())

history = model.fit(train_gen, validation_data=val_gen, epochs=5)
