import random

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertModel, BertTokenizerFast

alephbert = BertModel.from_pretrained('onlplab/alephbert-base')
alephbert_tokenizer = BertTokenizerFast.from_pretrained('onlplab/alephbert-base')
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
MAX_LEN = 64
BATCH_SIZE = 16
EPOCHS = 3

train_data = pd.read_csv('data/train_sentiment.csv')
test_data = pd.read_csv('data/test_sentiment.csv')

X_train = train_data.text.values
y_train = train_data.label.values
X_test = test_data.text.values
y_test = test_data.label.values
    
# Create a function to tokenize a set of texts
def preprocessing_for_bert(data):
    """Perform required preprocessing steps for pretrained BERT.
    @param    data (np.array): Array of texts to be processed.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    """
    # Create empty lists to store outputs
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in data:
        # `encode_plus` will:
        #    (1) Tokenize the sentence
        #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
        #    (3) Truncate/Pad sentence to max length
        #    (4) Map tokens to their IDs
        #    (5) Create attention mask
        #    (6) Return a dictionary of outputs
        encoded_sent = alephbert_tokenizer.encode_plus(
            text=sent,  # Preprocess sentence
            add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
            max_length=MAX_LEN,  # Max length to truncate/pad
            pad_to_max_length=True,  # Pad sentence to max length
            # return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True  # Return attention mask
        )

        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks

train_inputs, train_masks = preprocessing_for_bert(X_train)
val_inputs, val_masks = preprocessing_for_bert(X_test)

# Convert other data types to torch.Tensor
train_labels = torch.tensor(y_train)
val_labels = torch.tensor(y_test)

# Create the DataLoader for our training set
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

# Create the DataLoader for our validation set
val_data = TensorDataset(val_inputs, val_masks, val_labels)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=BATCH_SIZE)

# Specify loss function
loss_fn = nn.CrossEntropyLoss()


# Create the BertClassfier class
class BertClassifier(nn.Module):
    """Bert Model for Classification Tasks.
    """

    def __init__(self, freeze_bert=False):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(BertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, 50, 2

        # Instantiate BERT model
        self.bert = BertModel.from_pretrained('onlplab/alephbert-base')

        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(H, D_out)
        )

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (BATCH_SIZE, max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask information with shape (BATCH_SIZE, MAX_LEN)
        @return   logits (torch.Tensor): an output tensor with shape (BATCH_SIZE, num_labels)
        """
        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)

        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)

        return logits


def initialize_model(epochs=3):
    set_seed(42)
    """Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
    """
    # Instantiate Bert Classifier
    bert_classifier = BertClassifier(freeze_bert=False)
    bert_classifier.to(device)
    bert_classifier = torch.load("model", map_location=device)
    return bert_classifier


def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def predict(model, sample):
    val_inputs, val_masks = preprocessing_for_bert([sample])
    model.eval()
    with torch.no_grad():
        logits = model(val_inputs.to(device), val_masks.to(device))
    return F.softmax(logits).cpu().numpy()[0]


def predict_single_text(model: BertClassifier ,text: str, threshold: float = 0.7):
    prob = predict(model, text)
    if prob[1] > threshold:
        return "X"
    else:
        return "V"

def predict_single_text_with_norm(model: BertClassifier ,text: str, min_threshold: float = 0.3, max_threshold = 0.8):
    prob = predict(model, text)
    norm=0
    happy=1
    sad = 2
    if prob[1] > min_threshold:
        if prob[1]<max_threshold:
            return norm
        else:
            return sad
    else:
        return happy

def runme():
    # set_seed(42)
    bert_classifier = initialize_model(epochs=EPOCHS)
    text1 = "איזה איכסה לא כיף לי"
    text2 = "איזה יום היום"
    text3 = "אני אוכל דגנים"
    text4 = "אתה הנשיא הכי גרוע שהיה לנו רק פילגת והסתת"
    res = predict_single_text_with_norm(bert_classifier, text1, 0.3,0.8)
    print(res)
    res = predict_single_text_with_norm(bert_classifier, text2, 0.3,0.8)
    print(res)
    res = predict_single_text_with_norm(bert_classifier, text3, 0.3,0.8)
    print(res)
    res = predict_single_text_with_norm(bert_classifier, text4, 0.3,0.8)
    print(res)


# if __name__ == '__main__':
#     runme()