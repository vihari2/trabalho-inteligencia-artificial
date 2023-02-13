import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense

# Loading data from Names.txt and Wpc50.dat file
data = pd.read_csv("Wpc50.dat", sep='\s+', header=None, dtype=float, error_bad_lines=False)
species = pd.read_csv("Names.txt", header=None, names=["species"])

# print(data.head())
# print(data.describe())

# Encoding of the Names.txt file, the names are in string and I need to transform them to numerical data
encoder = LabelEncoder()
species_encoded = encoder.fit_transform(species.values.ravel())

# Classifying the species 
class_names = ['Morcego', 'Outro']
class_encoded = []
for i in species.species:
    if "Morcego" in i:
        class_encoded.append(1)
    else:
        class_encoded.append(0)

#Scaling of training data
scaler = StandardScaler()
data_train_normalized = scaler.fit_transform(data)

# Splitting the data into training and testing, 75% for training and 25% for testing.
data_train, data_test, class_train, class_test = train_test_split(data_train_normalized ,np.array(class_encoded) , test_size=0.25, random_state=0)

# Creating the model
model = Sequential()
model.add(Dense(128, input_dim=data_train.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compiling the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training the model
history = model.fit(data_train, class_train, epochs=10, batch_size=32, validation_data=(data_test, class_test))

# Evaluating the model
score = model.evaluate(data_test, class_test, verbose=0)
print("Accuracy:", score[1])
