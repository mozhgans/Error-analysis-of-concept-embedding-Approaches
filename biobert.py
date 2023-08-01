# -*- coding: utf-8 -*-


!pip install transformers

import torch
from transformers import BertTokenizer, BertModel

#bert-base-uncased model from Hugging Face model hub:
def get_bert_representations(text, model_name='bert-base-uncased'):
    # Load BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    # Tokenize the text and convert it to a tensor
    input_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])

    # Obtain BERT representations for the input tensor
    with torch.no_grad():
        model.eval()
        outputs = model(input_ids)
        representations = outputs.last_hidden_state

    return representations

# Example usage:
text = "Contextual word embeddings can enhance NLP tasks."
representations = get_bert_representations(text)
print(representations.shape)  # (1, num_tokens, hidden_size) - e.g., (1, 12, 768) for BERT-base-uncased

#biobert implementation; for this model, I used the dmis-lab/biobert-base-cased-v1.1 model, which is a pre-trained BioBERT model. You can replace it with other pre-trained BioBERT models available in the Hugging Face model hub.
import torch
from transformers import BertTokenizer, BertModel

def get_biobert_representations(text, model_name='dmis-lab/biobert-base-cased-v1.1'):
    # Load BioBERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    # Tokenize the text and convert it to a tensor
    input_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])

    # Obtain BioBERT representations for the input tensor
    with torch.no_grad():
        model.eval()
        outputs = model(input_ids)
        representations = outputs.last_hidden_state

    return representations

# Example usage:
text = "Contextual word embeddings can enhance NLP tasks."
representations = get_biobert_representations(text)
print(representations.shape)  # (1, num_tokens, hidden_size) - e.g., (1, 15, 768) for BioBERT-base-cased