import argparse, subprocess, sys

def pip_install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])
    
pip_install("transformers>=4.4.2")
pip_install("datasets[s3]==1.5.0")

import transformers
import datasets

from transformers import AutoTokenizer
from datasets import load_dataset

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    # preprocessing arguments
    parser.add_argument('--threshold', type=int, default=4)
    parser.add_argument('--s3-bucket', type=str)
    parser.add_argument('--s3-prefix', type=str)

    args, _ = parser.parse_known_args()
    print('Received arguments {}'.format(args))
    threshold = args.threshold
    s3_bucket = args.s3_bucket
    s3_prefix = args.s3_prefix

    # Download dataset
    train_dataset, valid_dataset, test_dataset = load_dataset('generated_reviews_enth', split=['train', 'validation', 'test'])
    print(train_dataset.shape)
    print(valid_dataset.shape)
    print(test_dataset.shape)
    
    # Replace star rating with 0-1 label
    def map_stars_to_sentiment(row):
        return {
            'labels': 1 if row['review_star'] >= threshold else 0
        }
    train_dataset = train_dataset.map(map_stars_to_sentiment)
    valid_dataset = valid_dataset.map(map_stars_to_sentiment)
    
    # Drop and rename columns
    train_dataset = train_dataset.flatten()
    valid_dataset = valid_dataset.flatten()
    
    train_dataset = train_dataset.remove_columns(['correct', 'translation.th', 'review_star'])
    valid_dataset = valid_dataset.remove_columns(['correct', 'translation.th', 'review_star'])

    train_dataset = train_dataset.rename_column('translation.en', 'text')
    valid_dataset = valid_dataset.rename_column('translation.en', 'text')
    
    # Tokenize data
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    
    def tokenize(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True)
    
    train_dataset = train_dataset.map(tokenize, batched=True, batch_size=len(train_dataset))
    valid_dataset = valid_dataset.map(tokenize, batched=True, batch_size=len(valid_dataset))
    
    # Drop the text variable, which we don't need anymore
    train_dataset = train_dataset.remove_columns(['text'])
    valid_dataset = valid_dataset.remove_columns(['text'])

    # Upload data to S3
    train_dataset.save_to_disk('/opt/ml/processing/output/training/')
    valid_dataset.save_to_disk('/opt/ml/processing/output/validation/')
    