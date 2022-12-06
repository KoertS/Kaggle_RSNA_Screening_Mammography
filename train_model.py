from model import create_model, pfbeta_tf
from train_val_split import get_train_val_generator

model = create_model()

train_gen, val_gen = get_train_val_generator()
metrics = [pfbeta_tf]
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=metrics)
print(model.summary())

history = model.fit(train_gen, validation_data=val_gen, epochs=2)
model.save('models/test_model.h5')
