# Honours_Project
Here is The Code for My honours Project

fdg.yml is the environment it was run in.
server.py	Sets up and runs the federated learning server.
split_and_save_data.py	Loads, cleans, scales, and splits the CICIDS2017 dataset into non-IID shards
client.py	Defines the federated learning client logic, including model creation, local training, optional gradient quantization, and communication with the server.
config.py	Stores global configuration values.
launch_clients.py	Automates the launch of multiple client processes in parallel, each representing a simulated federated client using the client.py logic.
AllData.py	Evaluates saved models from all clients by computing and displaying classification metrics. F1 score, AUC, confusion matrix, and ROC curves.
Scatter.py	Generates scatter plots of models trained with and without gradient quantization.
Testing_times.py	Measures and logs the evaluation time, loss, and accuracy of each client’s trained model, outputting the results to a CSV file.
