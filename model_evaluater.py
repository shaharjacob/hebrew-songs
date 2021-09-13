from click import secho
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, confusion_matrix


def evaluate_predection(labels, predicted_labels, kinds, average, title):
    prec_mac = precision_score(labels, predicted_labels, labels=kinds, average=average, zero_division=1)
    rec_mac = recall_score(labels, predicted_labels, labels=kinds, average=average)
    f1_mac = f1_score(labels, predicted_labels, labels=kinds, average=average)
    
    secho(f"{title}", fg="blue", bold=True)
    secho(f" precision: {prec_mac}, recall: {rec_mac}, f1: {f1_mac}", fg="blue")
    secho("---------------------------------------------------------")
    
    
def evaluate(predicted_labels, labels):
    evaluate_predection(labels, predicted_labels, None, 'micro', 'Micro')
    evaluate_predection(labels, predicted_labels, None, 'macro', 'Macro')
    acc = accuracy_score(labels, predicted_labels)
    cm = confusion_matrix(labels,predicted_labels)

    secho(f"Accuracy: {acc}", fg="blue")
    secho(f'The matrix header: {sorted(list(set(list(labels))))}', fg="blue")
    secho(f"Confusion matrix:", fg="blue")
    secho(f"{cm}", fg="blue")
