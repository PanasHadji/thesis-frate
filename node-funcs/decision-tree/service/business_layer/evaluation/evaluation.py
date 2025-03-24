from service.business_layer.dtos.model_data_dto import Dataset
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, confusion_matrix


class TreeEvaluator:
    def __init__(self):
        print('TreeEvaluator')

    def generate_performance(self, clf, data_inputs: Dataset):
        print('START generate_performance')

        print()
        print('START data_inputs.x_train')
        print(data_inputs.x_train)
        y_pred_train = clf.predict(data_inputs.x_train)

        print()
        print('START data_inputs.x_test')
        print(data_inputs.x_test)

        print()
        print('START data_inputs.y_test')
        print(data_inputs.y_test)
        print()
        print('START data_inputs.y_train')
        print(data_inputs.y_train)

        # Use predict_proba instead of predict
        print('START y_pred_test')
        y_pred_test = clf.predict_proba(data_inputs.x_test)  # Changed to predict_proba

        print('START Evaluate the model')
        # Evaluate the model
        print('START roc_auc')
        # For multi-class, ensure you're passing probabilities
        roc_auc = roc_auc_score(data_inputs.y_test, y_pred_test, multi_class='ovr')
        print('START precision')
        precision = precision_score(data_inputs.y_test, y_pred_test.argmax(axis=1), average='weighted')  # Using predicted class labels for precision
        print('START recall')
        recall = recall_score(data_inputs.y_test, y_pred_test.argmax(axis=1), average='weighted')  # Using predicted class labels for recall
        print('START accuracy')
        accuracy = accuracy_score(data_inputs.y_test, y_pred_test.argmax(axis=1))  # Using predicted class labels for accuracy
        print('START misclassification_rate')
        misclassification_rate = 1 - accuracy

        # Confusion Matrix
        print('START Confusion Matrix')
        cm = confusion_matrix(data_inputs.y_test, y_pred_test.argmax(axis=1))

        print('START Accurancy')
        train_accuracy = accuracy_score(data_inputs.y_train, y_pred_train)
        test_accuracy = accuracy_score(data_inputs.y_test, y_pred_test.argmax(axis=1))

        # Return evaluation metrics
        evaluation_results = {
            'Classifier': 'Decision Tree',
            'ROC AUC': roc_auc,
            'Precision': precision,
            'Recall': recall,
            'Accuracy': accuracy,
            'Misclassification Rate': misclassification_rate,
            'Training Accuracy': train_accuracy,
            'Testing Accuracy': test_accuracy,
            'Discrepancy in Accuracy': train_accuracy - test_accuracy
        }

        return evaluation_results
