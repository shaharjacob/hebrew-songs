from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, confusion_matrix


def evaluate_predection(labels,predicted_labels, kinds, average, title):
    prec_mac = precision_score(labels,
                               predicted_labels, labels=kinds, average=average,
                               zero_division=1)

    rec_mac = recall_score(labels,
                           predicted_labels, labels=kinds, average=average
                           )

    f1_mac = f1_score(labels,
                      predicted_labels, labels=kinds, average=average
                      )
    print(f"{title}")
    print(f" precision:{prec_mac}, recall:{rec_mac}, f1:{f1_mac}")
    print("---------------------------------------------------------")
def evaluate(predicted_labels, labels):
    evaluate_predection(labels, predicted_labels, None, 'micro', 'Micro')
    evaluate_predection(labels, predicted_labels, None, 'macro', 'Macro')

    acc = accuracy_score(labels,
                         predicted_labels)

    print(f"Accuracy: {acc}")
    cm = confusion_matrix(labels,predicted_labels)
    print('the matrix header: hit,no hit')
    print(cm)
