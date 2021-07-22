import boto3, argparse, subprocess, sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def pip_install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])
      
pip_install('sagemaker')
import sagemaker
from sagemaker.feature_store.feature_group import FeatureGroup

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    # preprocessing arguments
    parser.add_argument('--region', type=str)
    parser.add_argument('--bucket', type=str)

    args, _ = parser.parse_known_args()
    print('Received arguments {}'.format(args))
    region = args.region
    bucket = args.bucket

    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client(service_name='sagemaker')
    featurestore_client = boto_session.client(service_name='sagemaker-featurestore-runtime')
    session = sagemaker.session.Session(
        boto_session=boto_session, 
        sagemaker_client=sagemaker_client, 
        sagemaker_featurestore_runtime_client=featurestore_client)
    
    # Read feature group name
    with open('/opt/ml/processing/input/feature_group_name.txt') as f:
        feature_group_name = f.read()
    
    feature_group = FeatureGroup(name=feature_group_name, sagemaker_session=session)
    
    feature_group_query = feature_group.athena_query()
    feature_group_table = feature_group_query.table_name
    print(feature_group_table)
    
    query_string = 'SELECT label,review_body FROM "' \
        + feature_group_table+'"' \
        + ' INNER JOIN (SELECT product_id FROM (SELECT product_id, avg(star_rating) as avg_rating, count(*) as review_count \
            FROM "' + feature_group_table+'"' \
        + ' GROUP BY product_id) WHERE review_count > 1000) tmp ON "' \
        + feature_group_table+'"'+ '.product_id=tmp.product_id;'
    print(query_string)
    
    dataset = pd.DataFrame()
    feature_group_query.run(
        query_string=query_string, 
        output_location='s3://'+bucket+'/query_results/')
    feature_group_query.wait()
    
    dataset = feature_group_query.as_dataframe()
    dataset.head()
    
    training, validation = train_test_split(dataset, test_size=0.1)
    print(training.shape)
    print(validation.shape)
    
    np.savetxt('/opt/ml/processing/output/training/training.txt', training.values, fmt='%s')
    np.savetxt('/opt/ml/processing/output/validation/validation.txt', validation.values, fmt='%s')
    
    print('Job complete')
