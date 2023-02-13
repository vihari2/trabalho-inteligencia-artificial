import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# Loading data from Names.txt and Wpc50.dat file
data = pd.read_csv("Wpc50.dat", sep='\s+', header=None, dtype=float)
species = pd.read_csv("Names.txt", header=None, names=["species"])

#Scaling of training data
scaler = StandardScaler()
data_train_normalized = scaler.fit_transform(data)

# Encoding of the Names.txt file, the names are in string and I need to transform them to numerical data
encoder = LabelEncoder()
species_encoded = encoder.fit_transform(species.values.ravel())

# Splitting the data into training and testing, 75% for training and 25% for testing.
data_train, data_test, species_train, species_test = train_test_split(data_train_normalized, species_encoded, test_size=0.25, random_state=0)

# Converting the target variables to categorical
species_train = np.array(species_train)
species_train = np.eye(np.max(species_train) + 1)[species_train]
species_test = np.array(species_test)
species_test = np.eye(np.max(species_test) + 1)[species_test]

# Defining the model
model = Sequential()
model.add(Dense(1, input_dim=data_train.shape[1], activation='sigmoid'))
model.add(Dense(species_train.shape[1], activation='softmax'))

# Compiling the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training the model
model.fit(data_train, species_train, epochs=1000, batch_size=1, verbose=0)

# Evaluating the model
score = model.evaluate(data_test, species_test, verbose=0)
print("Accuracy:", score[1])



