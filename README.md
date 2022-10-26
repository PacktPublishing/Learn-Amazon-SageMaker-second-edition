# Learn Amazon SageMaker - Second Edition

<a href="https://www.packtpub.com/product/learn-amazon-sagemaker-second-edition/9781801817950?utm_source=github&utm_medium=repository&utm_campaign=9781801817950"><img src="https://static.packt-cdn.com/products/9781801817950/cover/smaller" alt="Learn Amazon SageMaker, Second Edition" height="256px" align="right"></a>

This is the code repository for [Learn Amazon SageMaker - Second Edition](https://www.packtpub.com/product/learn-amazon-sagemaker-second-edition/9781801817950?utm_source=github&utm_medium=repository&utm_campaign=9781801817950), published by Packt.

**A guide to building, training, and deploying machine learning models for developers and data scientists**

## What is this book about?
This updated second edition of Learn Amazon SageMaker will teach you how to move quickly from business questions to high performance models in production. Using machine learning and deep learning examples implemented with Python and Jupyter notebooks, you’ll learn how to make the most of the many features and APIs of Amazon SageMaker.	

This book covers the following exciting features: 
* Become well-versed with data annotation and preparation techniques
* Use AutoML features to build and train machine learning models with AutoPilot
* Create models using built-in algorithms and frameworks and your own code
* Train computer vision and natural language processing (NLP) models using real-world examples
* Cover training techniques for scaling, model optimization, model debugging, and cost optimization
* Automate deployment tasks in a variety of configurations using SDK and several automation tools

If you feel this book is for you, get your [copy](https://www.amazon.com/dp/1801077053) today!

<a href="https://www.packtpub.com/?utm_source=github&utm_medium=banner&utm_campaign=GitHubBanner"><img src="https://raw.githubusercontent.com/PacktPublishing/GitHub/master/GitHub.png" 
alt="https://www.packtpub.com/" border="5" /></a>


## Instructions and Navigations
All of the code is organized into folders.

The code will look like the following:
```
od = sagemaker.estimator.Estimator(
container,
role,
train_instance_count=2,
train_instance_type='ml.p3.2xlarge',
train_use_spot_instances=True,
train_max_run=3600, # 1 hours
train_max_wait=7200, # 2 hour
output_path=s3_output)
```

**Following is what you need for this book:**
This book is for software engineers, machine learning developers, data scientists, and AWS users who are new to using Amazon SageMaker and want to build high-quality machine learning models without worrying about infrastructure. Knowledge of AWS basics is required to grasp the concepts covered in this book more effectively. A solid understanding of machine learning concepts and the Python programming language will also be beneficial.	

### Software and Hardware List

You will need a functional AWS account for running everything.


We also provide a PDF file that has color images of the screenshots/diagrams used in this book. [Click here to download it](https://static.packt-cdn.com/downloads/9781801817950_ColorImages.pdf).


### Related products <Other books you may enjoy>
* Machine Learning with Amazon SageMaker Cookbook [[Packt]](https://www.packtpub.com/product/machine-learning-with-amazon-sagemaker-cookbook/9781800567030?utm_source=github&utm_medium=repository&utm_campaign=9781800567030) [[Amazon]](https://www.amazon.com/dp/B093TJ9KG2)

* Amazon Redshift Cookbook [[Packt]](https://www.packtpub.com/product/amazon-redshift-cookbook/9781800569683?utm_source=github&utm_medium=repository&utm_campaign=9781800569683) [[Amazon]](https://www.amazon.com/dp/1800569688)

## Get to Know the Author
**Julien Simon**
is a Principal Developer Advocate for AI & Machine Learning at Amazon Web Services. He focuses on helping developers and enterprises bring their ideas to life. He frequently speaks at conferences, blogs on the AWS Blog and on Medium, and he also runs an AI/ML podcast.
Prior to joining AWS, Julien served for 10 years as CTO/VP Engineering in top-tier web startups where he led large Software and Ops teams in charge of thousands of servers worldwide. In the process, he fought his way through a wide range of technical, business and procurement issues,
which helped him gain a deep understanding of physical infrastructure, its limitations and how cloud computing can help.
### Download a free PDF

 <i>If you have already purchased a print or Kindle version of this book, you can get a DRM-free PDF version at no cost.<br>Simply click on the link to claim your free PDF.</i>
<p align="center"> <a href="https://packt.link/free-ebook/9781801817950">https://packt.link/free-ebook/9781801817950 </a> </p>