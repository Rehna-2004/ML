from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score  

#Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

#Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
# Create KNN classifier with k=3
knn = KNeighborsClassifier(n_neighbors=3)
# Train the classifier
knn.fit(X_train, y_train)
#predict the test data
y_pred = knn.predict(X_test)
#find the accuracy
accuracy = accuracy_score(y_test, y_pred)
# Display the results
print("Actual Values:",y_test)
print("Predicted Values:",y_pred)
print("Accuracy :",accuracy * 100,"%")
                                               
