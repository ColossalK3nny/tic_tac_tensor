from tensorflow.keras import layers, models

def create_model():
    model = models.Sequential([
        layers.Input(shape=(9,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(9, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model