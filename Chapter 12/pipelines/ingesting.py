import boto3, argparse, os, subprocess, sys
import time
from time import time, gmtime, strftime, sleep
import pandas as pd

def pip_install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])
      
pip_install('sagemaker')
import sagemaker

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    # preprocessing arguments
    parser.add_argument('--region', type=str)
    parser.add_argument('--bucket', type=str)
    parser.add_argument('--prefix', type=str, default='amazon-reviews-featurestore')
    parser.add_argument('--role', type=str)
    parser.add_argument('--feature-group-name', type=str)
    parser.add_argument('--max-workers', type=int, default=4)


    args, _ = parser.parse_known_args()
    print('Received arguments {}'.format(args))
    region = args.region
    bucket = args.bucket
    prefix = args.prefix
    role = args.role
    fg_name = args.feature_group_name
    max_workers = args.max_workers

    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client(service_name='sagemaker')
    featurestore_client = boto_session.client(service_name='sagemaker-featurestore-runtime')
    session = sagemaker.session.Session(
        boto_session=boto_session, 
        sagemaker_client=sagemaker_client, 
        sagemaker_featurestore_runtime_client=featurestore_client)
    
    # Load input data
    input_data_path = '/opt/ml/processing/input/fs_data.tsv'
    print('Reading input data from {}'.format(input_data_path))
    data = pd.read_csv(input_data_path, sep='\t',
                       error_bad_lines=False, dtype='str')
    
    # Define the feature group name
    print('Creating feature group...')
    from sagemaker.feature_store.feature_group import FeatureGroup
    feature_group = FeatureGroup(name=fg_name, sagemaker_session=session)
    # Define the name of the column storing a unique record id (e.g. primary key)
    record_identifier_feature_name = 'review_id'
    # Add a column to store feature timestamps
    event_time_feature_name = 'event_time'
    current_time_sec = int(round(time()))
    data = data.assign(event_time=current_time_sec)
    # Set the correct type for each column
    data['review_id']     = data['review_id'].astype('str').astype('string')
    data['product_id']    = data['product_id'].astype('str').astype('string')
    data['review_body']   = data['review_body'].astype('str').astype('string')
    data['label']         = data['label'].astype('str').astype('string')
    data['star_rating']   = data['star_rating'].astype('int64')
    data['event_time']    = data['event_time'].astype('float64')
    # Load feature definitions
    feature_group.load_feature_definitions(data_frame=data)
    # Create feature group
    feature_group.create(
        s3_uri='s3://{}/{}'.format(bucket, prefix),
        record_identifier_name=record_identifier_feature_name,
        event_time_feature_name=event_time_feature_name,
        role_arn=role,
        enable_online_store=True,
        description="1.8M+ tokenized camera reviews from the Amazon Customer Reviews dataset",
        tags=[
            { 'Key': 'Dataset', 'Value': 'amazon customer reviews' },
            { 'Key': 'Subset', 'Value': 'cameras' },
            { 'Key': 'Owner', 'Value': 'Julien Simon' }
        ]
    )
    # Wait for feature group to be ready
    while feature_group.describe().get("FeatureGroupStatus") != 'Created':
        sleep(1)
    print('Feature group created')
    
    # Ingest data
    print('Ingesting data...')
    try:
        feature_group.ingest(data_frame=data, max_workers=max_workers, wait=True)
    except Exception:
        pass
    
    print('Waiting...')
    # Wait for 10 minutes to make sure data has flowed to the offline store
    # https://docs.aws.amazon.com/sagemaker/latest/dg/feature-store-offline.html
    sleep(600)

    # Save feature group name
    with open('/opt/ml/processing/output/feature_group_name.txt', 'w+') as f:
        f.write(fg_name)
    
    print('Job complete')