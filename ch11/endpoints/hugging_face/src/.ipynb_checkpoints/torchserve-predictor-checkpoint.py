import json
import torch
from transformers import AutoConfig, AutoTokenizer, DistilBertForSequenceClassification

JSON_CONTENT_TYPE = 'application/json'
CLASS_NAMES = ['negative', 'positive']

def model_fn(model_dir):
    config_path = '{}/config.json'.format(model_dir)
    model_path =  '{}/pytorch_model.bin'.format(model_dir)
    config = AutoConfig.from_pretrained(config_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path, config=config)
    print(config)
    return model

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

def predict_fn(input_data, model):
    inputs = tokenizer(input_data['text'], return_tensors='pt')
    outputs = model(**inputs)
    logits = outputs.logits
    _, prediction = torch.max(logits, dim=1)
    return CLASS_NAMES[prediction]

def input_fn(serialized_input_data, content_type=JSON_CONTENT_TYPE):  
    if content_type == JSON_CONTENT_TYPE:
        input_data = json.loads(serialized_input_data)
        return input_data
    else:
        raise Exception('Unsupported input type: ' + content_type)

def output_fn(prediction_output, accept=JSON_CONTENT_TYPE):
    if accept == JSON_CONTENT_TYPE:
        return json.dumps(prediction_output), accept
    else:
        raise Exception('Unsupported output type: ' + accept)
