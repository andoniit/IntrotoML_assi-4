


################ Problem 1: Clustering ###############################
'''
For this task, you will perform various clustering-related opera:ons over datasets lab04_dataset_1.csv and lab04_dataset_2.csv, using sklearn’s clustering module.
1. Dataset lab04_dataset_1.csv have two variables x1 and x2. Apply KMeans algorithm on the two-dimensional data and output the resul:ng clusters using a scaGerplot. You will apply KMeans over several clusters ranging from cluster-count K = 2 to 6. Make sure for every itera:on of different cluster-count, your scaGerplot should use K colors to clearly dis:nguish the data points belonging in their respec:ve K clusters. Also, compute the SilhoueGe score for each of those K clusters and plot that score against K. (5 marks)
2. Dataset lab04_dataset_2.csv have two variables x1 and x2. Again, apply KMeans algorithm on the two-dimensional data with clusters ranging from K = 2 to 4 and output the resul:ng clusters using scaGerplots. Do the cluster outputs you obtained using KMeans for this dataset make sense? (5 marks)
3. The data in the lab04_dataset_2.csv forms 4 concentric rings rather than being well- separated clusters. So ideally, we would want 4 clusters represen:ng the 4 concentric rings. KMeans is not well-suited to handle data like this. Use SpectralClustering to cluster the data. Show the results for clusters K = 2 to 4 (5 marks)
4. It is possible that SpectralClustering although an improvement over KMeans is s:ll not able to create 4 clusters corresponding to the 4 concentric rings. Explore the other sklearn clustering algorithms to see which one can produce 4 clusters corresponding with the 4
    
concentric rings. Hint: I men:oned this algorithm during our class while discussing density-based clustering. (5 marks)

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from sklearn.metrics import silhouette_score


# Load datasets
data1 = pd.read_csv('lab04_dataset_1.csv')
data2 = pd.read_csv('lab04_dataset_2.csv')

################ Problem 1.1: Clustering ###############################

# Task 1: Clustering lab04_dataset_1.csv using KMeans
def cluster_dataset1():
    X1 = data1[['x1', 'x2']]
    k_range = range(2, 7)
    plt.figure(figsize=(20, 5))

    for i, k in enumerate(k_range, 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X1)
        labels = kmeans.labels_
        silhouette_avg = silhouette_score(X1, labels)

        plt.subplot(1, len(k_range), i)
        plt.scatter(X1['x1'], X1['x2'], c=labels, cmap='viridis')
        plt.title(f'K = {k}, Silhouette Score: {silhouette_avg:.2f}')
        plt.xlabel('x1')
        plt.ylabel('x2')
        #plt.colorbar()

   # plt.tight_layout()
    plt.show()





# Extract features
X1 = data1[['x1', 'x2']]

# Define range of cluster counts
k_range = range(2, 7)

# Initialize lists to store silhouette scores
silhouette_scores = []

# Iterate over different cluster counts
for k in k_range:
    # Initialize KMeans with current cluster count
    kmeans = KMeans(n_clusters=k, random_state=42)
    
    # Fit KMeans to the data
    kmeans.fit(X1)
    
    # Predict cluster labels
    labels = kmeans.labels_
    
    # Compute Silhouette score
    silhouette_avg = silhouette_score(X1, labels)
    
    # Append silhouette score to list
    silhouette_scores.append(silhouette_avg)

# Plot Silhouette scores against K
plt.plot(k_range, silhouette_scores, marker='o')
plt.title('Silhouette Score vs. Number of Clusters')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.xticks(k_range)
plt.grid(True)
plt.show()

################ Problem 1.2: Clustering ###############################

# Task 2: Clustering lab04_dataset_2.csv using KMeans
def cluster_dataset2_kmeans():
    X2 = data2[['x1', 'x2']]
    k_range = range(2, 5)
    plt.figure(figsize=(15, 5))

    for i, k in enumerate(k_range, 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X2)
        labels = kmeans.labels_

        plt.subplot(1, len(k_range), i)
        plt.scatter(X2['x1'], X2['x2'], c=labels, cmap='viridis')
        plt.title(f'KMeans Clustering with K = {k}')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.colorbar()

    #plt.tight_layout()
    plt.show()


################ Problem 1.3: Clustering ###############################

    # Task 3: Clustering lab04_dataset_2.csv using SpectralClustering
def cluster_dataset2_spectral():
    X2 = data2[['x1', 'x2']]
    k_range = range(2, 5)
    plt.figure(figsize=(15, 5))

    for i, k in enumerate(k_range, 1):
        spectral = SpectralClustering(n_clusters=k, random_state=42)
        labels = spectral.fit_predict(X2)

        plt.subplot(1, len(k_range), i)
        plt.scatter(X2['x1'], X2['x2'], c=labels, cmap='viridis')
        plt.title(f'Spectral Clustering with K = {k}')
        plt.xlabel('x1')
        plt.ylabel('x2')

    #plt.tight_layout()
    plt.show()

################ Problem 1.4: Clustering ###############################


# Task 4: Clustering lab04_dataset_2.csv using DBSCAN
def cluster_dataset2_dbscan():
    X2 = data2[['x1', 'x2']]
    dbscan = DBSCAN(eps=0.1, min_samples=5)
    labels = dbscan.fit_predict(X2)

    plt.figure(figsize=(8, 6))
    plt.scatter(X2['x1'], X2['x2'], c=labels, cmap='viridis')
    plt.title('DBSCAN Clustering')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.colorbar()
    #plt.tight_layout()
    plt.show()

# Perform clustering for each task
cluster_dataset1()
cluster_dataset2_kmeans()
cluster_dataset2_spectral()
cluster_dataset2_dbscan()

################ Problem 2: Neural Network ###############################


'''
For this task, you will perform various neural network-related opera:ons over datasets lab04_dataset_3.csv and lab04_dataset_4.csv, using sklearn’s neural network module.
1. You will train a Mul:-Layer Perceptron neural network for the task of classifica:on on the dataset lab04_dataset_3.csv using MLPClassifier. The inputs to your MLPClassifier are alcohol, citric_acid, free_sulfur_dioxide, residual_sugar, sulphates, while the output is quality_grp, which has two categories, namely, 0 and 1. Use a train-test split of 80-20. For the learning task, you will train neural network models with different architectures:
a. Ac:va:on func:on = [logis:c, relu, tanh]
b. Hidden layers = [1, 2, 3, 4, 5]
c. Neurons per layer = [2, 4, 6, 8]
So, basically in the first itera:on you will create a learning model using the neural network architecture [logis:c, 1, 2], in the second itera:on you will use [logis:c, 1, 4], all the way to [tanh, 5, 8]. For each of these learned models, compute the Misclassifica:on Rate on the test set. Once you are done with all the training, you should output a dataframe with columns Ac9va9on func9on, Hidden layers, Neurons per layer, Misclassifica9on Rate, where each row will correspond with the individual training models. Since the total count of ac:va:on func:ons, hidden layers and neurons are 3, 5, 4 respec:vely, the number of rows in your output dataframe should be 3 x 5 x 4 = 60. Also use max_iter=10000 and random_state=2023484 inside MLPClassifier defini:on (10 marks)
2. Create a scaGerplot of the Misclassifica:on Rate, make sure that the misclassifica:on rates are dis:nguishable by different colors according to their ac:va:on func:on. So, the scaGerplot should have 3 colors differen:a:ng the misclassifica:on rates associated with the 3 ac:va:on func:ons. (3 marks)
3. The model with the lowest Misclassifica:on Rate is the best neural network. Output the model parameters (ac:va:on func:on, hidden layers, neurons) of this neural network. In the case of :es, choose the network with fewer neurons overall. (2 marks)

'''
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import seaborn as sea

dataset_3 = pd.read_csv("lab04_dataset_3.csv")

X = dataset_3[["alcohol", "citric_acid", "free_sulfur_dioxide", "residual_sugar", "sulphates"]]
y = dataset_3["quality_grp"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

activation_functions = ["logistic", "relu", "tanh"]
hidden_layers = [1, 2, 3, 4, 5]
neurons_per_layer = [2, 4, 6, 8]

results = []

for activation_function in activation_functions:
    for hidden_layer in hidden_layers:
        for neuron_per_layer in neurons_per_layer:
            clf = MLPClassifier(hidden_layer_sizes=(neuron_per_layer,) * hidden_layer, activation=activation_function,
                                max_iter=10000, random_state=2023484)
            
            clf.fit(X_train, y_train)
            
            misclassification_rate = 1 - clf.score(X_test, y_test)
            
            results.append({"Activation Function": activation_function,
                            "Hidden Layers": hidden_layer,
                            "Neurons per Layer": neuron_per_layer,
                            "Misclassification Rate": misclassification_rate})

results_df = pd.DataFrame(results)
print(results_df)


#Plot the graph

plt.figure(figsize=(10, 6))
sea.scatterplot(data=results_df, x=range(len(results_df)), y='Misclassification Rate', hue='Activation Function', palette='Set1')
plt.title('Misclassification Rate Graph')
plt.xlabel('Index')
plt.ylabel('Misclassification Rate')
plt.legend(title="Activation Function")
plt.grid(True)

#print the results

sorted_results_df = results_df.sort_values(by="Misclassification Rate")

best_model = sorted_results_df.iloc[0]

print("Best Model Parameters:")
print("Activation Function:", best_model["Activation Function"])
print("Hidden Layers:", best_model["Hidden Layers"])
print("Neurons per Layer:", best_model["Neurons per Layer"])
plt.show()


'''
4. YouwilltrainaMul:-LayerPerceptronneuralnetworkforthetaskofregressiononthe dataset lab04_dataset_4.csv using MLPRegressor. The inputs to your MLPRegressor are housing_median_age, total_rooms, households, median_income and the output is median_house_value. First, normalize the dataset, and then create an 80-20 train-test split. In a similar manner to the previous classifica:on task, you will once again learn mul:ple neural network models of varying architectures.
a. Ac:va:on func:on = [relu, tanh]
b. Hidden layers = [2, 3, 4]
c. Neurons per layer = [4, 6, 8]
For each of these learned models, compute the Root Mean Square Error. Once you are done with all the training, you should output a dataframe with columns Ac9va9on func9on, Hidden layers, Neurons per layer, Root Mean Square Error, where each row will correspond with the individual training models. Since the total count of ac:va:on func:ons, hidden layers and neurons are 2, 3, 3 respec:vely, the number of rows in your output dataframe should be 2 x 3 x 3 = 18. Also use random_state=2023484 inside MLPRegressor defini:on (10 marks)
5. Create a scaGerplot of the Root Mean Square Error, make sure that the root mean square errors are dis:nguishable by different colors according to their ac:va:on func:on. So, the scaGerplot should have 2 colors differen:a:ng the root mean square errors associated with the 2 ac:va:on func:ons. (3 marks)
6. The model with the lowest Root Mean Square Error is the best neural network. Output the model parameters (ac:va:on func:on, hidden layers, neurons) of this neural network. In the case of :es, choose the network with fewer neurons overall. (2 marks)


'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset for regression
data_path_regression = "lab04_dataset_4.csv"
try:
    data_regression = pd.read_csv(data_path_regression)
except FileNotFoundError:
    print("File not found. Please check the path and try again.")
    exit()

# Select the inputs and the output for regression
X_regression = data_regression[['housing_median_age', 'total_rooms', 'households', 'median_income']]
y_regression = data_regression['median_house_value']

# Normalize the input features (X data)
scaler_X = StandardScaler()
X_regression_scaled = scaler_X.fit_transform(X_regression)

# Normalize the output target (y data) using Min-Max scaling
scaler_y = MinMaxScaler()
y_regression_scaled = scaler_y.fit_transform(y_regression.values.reshape(-1, 1)).flatten()

# Splitting the data into train and test sets for regression
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_regression_scaled, y_regression_scaled, test_size=0.2, random_state=2023484)

# Define the parameters for MLPRegressor
activation_functions_reg = ['relu', 'tanh']
hidden_layers_options_reg = [2, 3, 4]
neurons_per_layer_options_reg = [4, 6, 8]

# List to store results for regression
results_reg = []

# Train models with varying parameters for regression
for activation in activation_functions_reg:
    for layers in hidden_layers_options_reg:
        for neurons in neurons_per_layer_options_reg:
            hidden_layer_sizes = tuple([neurons] * layers)
            mlp_reg = MLPRegressor(activation=activation, hidden_layer_sizes=hidden_layer_sizes, random_state=2023484)
            try:
                mlp_reg.fit(X_train_reg, y_train_reg)
            except ValueError as e:
                print(f"An error occurred: {e}")
                continue
            y_pred_reg = mlp_reg.predict(X_test_reg)
            rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
            results_reg.append({
                "Activation function": activation,
                "Hidden layers": layers,
                "Neurons per layer": neurons,
                "Root Mean Square Error": rmse
            })

# Create a DataFrame to hold the regression results
results_df_reg = pd.DataFrame(results_reg)

# Find the model with the lowest Root Mean Square Error
best_model_reg = results_df_reg.loc[results_df_reg['Root Mean Square Error'].idxmin()]

# Plotting the root mean square errors
fig, ax = plt.subplots()
for activation in activation_functions_reg:
    subset = results_df_reg[results_df_reg['Activation function'] == activation]
    ax.scatter(subset['Hidden layers'] * subset['Neurons per layer'], subset['Root Mean Square Error'], label=activation)
ax.set_xlabel('Total Neurons')
ax.set_ylabel('Root Mean Square Error')
ax.set_title('Root Mean Square Error by Model Configuration')
ax.legend(title='Activation Function')
plt.show()

print(results_df_reg)
print("Best Model:", best_model_reg)
