import argparse, os, subprocess, sys
from time import gmtime, strftime

import pandas as pd
import numpy as np

import boto3

def pip_install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])
    
def spacy_install(package):
    subprocess.call([sys.executable, "-m", "spacy", "download", package])
    
if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    # preprocessing arguments
    parser.add_argument('--filename', type=str)
    parser.add_argument('--num-reviews', type=int)
    parser.add_argument('--library', type=str, default='spacy')

    args, _ = parser.parse_known_args()
    print('Received arguments {}'.format(args))
    filename = args.filename
    num_reviews = args.num_reviews
    library = args.library

    # Load dataset into a pandas dataframe
    input_data_path = os.path.join('/opt/ml/processing/input', filename)
    print('Reading input data from {}'.format(input_data_path))
    data = pd.read_csv(input_data_path, sep='\t', compression='gzip',
                       error_bad_lines=False, dtype='str')
    
    # Remove lines with missing values
    data.dropna(inplace=True)
    
    # Keep only 'num_reviews' rows
    if num_reviews is not None:
        data = data[:num_reviews]
        
    # Drop unwanted columns
    data['review_body'] = data['review_headline'] + ' ' + data['review_body']
    data = data[['review_id', 'product_id', 'star_rating', 'review_body']]
        
    # Add label column
    data['label'] = data.star_rating.map({
        '1': '__label__negative__',
        '2': '__label__negative__',
        '3': '__label__neutral__',
        '4': '__label__positive__',
        '5': '__label__positive__'})
    
    # Tokenize data
    print('Tokenizing reviews')
    
    if library == 'nltk':
        pip_install('nltk')
        import nltk
        nltk.download('punkt')
        data['review_body'] = data['review_body'].apply(nltk.word_tokenize)
        data['review_body'] = data.apply(lambda row: " ".join(row['review_body']).lower(), axis=1)
        
    elif library == 'spacy':
        pip_install('spacy')
        spacy_install('en_core_web_sm')
        import spacy
        spacy_nlp = spacy.load('en_core_web_sm')

        def tokenize(text):
            tokens = spacy_nlp.tokenizer(text)
            tokens = [ t.text for t in tokens ]
            return " ".join(tokens).lower()
        data['review_body'] = data['review_body'].apply(tokenize)
    
    else:
        print('Incorrect library name: should be nltk or spacy.')
        exit()
    
    # Create output dirs
    bt_output_dir = '/opt/ml/processing/output/bt/'
    fs_output_dir = '/opt/ml/processing/output/fs/'
    os.makedirs(bt_output_dir, exist_ok=True)
    os.makedirs(fs_output_dir, exist_ok=True)
    
    # Save data in TSV format for SageMaker Feature Store 
    fs_output_path = os.path.join(fs_output_dir, 'fs_data.tsv')    
    
    print('Saving SageMaker Feature Store training data to {}'.format(fs_output_path))
    data.to_csv(fs_output_path, index=False, header=True, sep='\t')

    # Save data in BlazingText format, with label column at the front
    bt_output_path = os.path.join(bt_output_dir, 'bt_data.txt')    
    
    data = data[['label', 'review_body']]
    print('Saving BlazingText data to {}'.format(bt_output_path))
    np.savetxt(bt_output_path, data.values, fmt='%s')