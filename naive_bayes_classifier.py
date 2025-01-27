import pandas as pd
import numpy as np


file_path = "dataset.csv"
dataset = pd.read_csv(file_path)

# log file
log_file = "classification_log.txt"

# print dataset also store it in the log file
print("Dataset:")
print(dataset)

with open(log_file, "w") as log:
    log.write("Classification Log\n")
    log.write("="*50 + "\n")
    # print dataset into the log file
    log.write("Dataset:\n")
    log.write(str(dataset) + "\n")
    
def calculate_probabilities(training_set):
    total_instances = len(training_set)
    yes_instances = len(training_set[training_set['PlayTennis'] == 'Yes'])
    no_instances = len(training_set[training_set['PlayTennis'] == 'No'])
    
    # prior probabilities
    prob_yes = yes_instances / total_instances
    prob_no = no_instances / total_instances
    
    # print operation done above into the log file
    with open(log_file, "a") as log:
        log.write(f"\nPrior Probabilities:\n")
        log.write(f"  P(Yes): {prob_yes}\n")
        log.write(f"  P(No): {prob_no}\n")

    # likelihood or conditional probabilities with Laplace smoothing
    features = ['Outlook', 'Temperature', 'Humidity', 'Wind']
    likelihood_probabilities = []

    for feature in features:
        # unique feature values in the entire dataset used for Laplace smoothing
        unique_values = dataset[feature].unique()
        num_unique = len(unique_values)

        # counts of feature values for each class
        counts_yes = training_set[training_set['PlayTennis'] == 'Yes'][feature].value_counts()
        counts_no = training_set[training_set['PlayTennis'] == 'No'][feature].value_counts()
        
        # smoothing and normalization
        probabilities_yes = {
            value: (counts_yes.get(value, 0) + 1) / (yes_instances + num_unique)
            for value in unique_values
        }
        probabilities_no = {
            value: (counts_no.get(value, 0) + 1) / (no_instances + num_unique)
            for value in unique_values
        }

        likelihood_probabilities.append({
            "feature": feature,
            "probabilities_yes": probabilities_yes,
            "probabilities_no": probabilities_no
        })

    return np.log(prob_yes), np.log(prob_no), likelihood_probabilities  

def predict(instance, log_prob_yes, log_prob_no, likelihood_probabilities):
    # initialization of posterior probabilities
    log_posterior_yes = log_prob_yes
    log_posterior_no = log_prob_no

    for likelihood in likelihood_probabilities:
        feature = likelihood['feature']
        value = instance[feature]
        log_posterior_yes += np.log(likelihood['probabilities_yes'].get(value, 1e-9)) # 1e-9 is the smoothing factor
        log_posterior_no += np.log(likelihood['probabilities_no'].get(value, 1e-9))

    return "Yes" if log_posterior_yes > log_posterior_no else "No"

# performance metrics initialization
true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0
accuracies = []

# train on the entire dataset
log_prob_yes, log_prob_no, likelihood_probabilities = calculate_probabilities(dataset)

# print likelihood probabilities into files
with open(log_file, "a") as log:
    log.write(f"\nLikelihoods for the entire dataset:\n")
    for likelihood in likelihood_probabilities:
        log.write(f"Feature: {likelihood['feature']}\n")
        log.write(f"  Probabilities Yes: {likelihood['probabilities_yes']}\n")
        log.write(f"  Probabilities No: {likelihood['probabilities_no']}\n")

# test on the entire dataset
for index, test_instance in dataset.iterrows():
    # predict the class for the test instance
    prediction = predict(test_instance, log_prob_yes, log_prob_no, likelihood_probabilities)
    
    is_correct = int(prediction == test_instance['PlayTennis'])
    # append the classification result to the list of accuracies
    accuracies.append(is_correct)

    # update confusion matrix
    actual = test_instance['PlayTennis']
    if actual == "Yes" and prediction == "Yes":
        true_positives += 1
    elif actual == "Yes" and prediction == "No":
        false_negatives += 1
    elif actual == "No" and prediction == "No":
        true_negatives += 1
    elif actual == "No" and prediction == "Yes":
        false_positives += 1

    with open(log_file, "a") as log:
        log.write(f"\nInstance {index + 1}:\n")
        log.write(f"  Actual Class: {actual}\n")
        log.write(f"  Predicted Class: {prediction}\n")
        log.write(f"  Correct Prediction: {is_correct}\n")

accuracy = sum(accuracies) / len(accuracies)
confusion_matrix = {
    "True Positives": true_positives,
    "False Positives": false_positives,
    "True Negatives": true_negatives,
    "False Negatives": false_negatives
}

# final results with performance metrics
with open(log_file, "a") as log:
    log.write("\nFinal Metrics:\n")
    log.write(f"Accuracy: {accuracy:.2f}\n")
    log.write(f"Confusion Matrix: {confusion_matrix}\n")

print(f"\nAccuracy: {accuracy:.2f}")
print("Confusion Matrix:")
for key, value in confusion_matrix.items():
    print(f"  {key}: {value}")
