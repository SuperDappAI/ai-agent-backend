import os


from dotenv import load_dotenv
import boto3
from boto3.dynamodb.conditions import Key
import json
import zipfile
import io
import time
import pytz
import aioboto3
from fastapi import APIRouter
from pydantic import BaseModel
from datetime import datetime, timedelta
from botocore.exceptions import ClientError

router = APIRouter()

# Load environment variables
load_dotenv()
AWS_ACCOUNT_ID = os.getenv("AWS_ACCOUNT_ID")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")
executor_role_name = 'interpreter-executor-role'
admin_role_name = 'interpreter-admin-role'
# Initialize AWS SDK
lambda_client = boto3.client('lambda')
dynamodb = boto3.resource('dynamodb')
iam = boto3.client('iam')
efs_client = boto3.client('efs')
ec2 = boto3.resource('ec2')

credentials = None
expiration = None
subnet_ids = []
security_group_ids = []
def update_clients_with_credentials(creds):
    global lambda_client, dynamodb, efs_client, ec2
    lambda_client = boto3.client('lambda',  
        aws_access_key_id=creds['AccessKeyId'],
        aws_secret_access_key=creds['SecretAccessKey'],
        aws_session_token=creds['SessionToken']
    )
    dynamodb = boto3.resource('dynamodb',  
        aws_access_key_id=creds['AccessKeyId'],
        aws_secret_access_key=creds['SecretAccessKey'],
        aws_session_token=creds['SessionToken']
    )
    efs_client = boto3.client('efs',  
        aws_access_key_id=creds['AccessKeyId'],
        aws_secret_access_key=creds['SecretAccessKey'],
        aws_session_token=creds['SessionToken']
    )
    ec2 = boto3.resource('ec2',  
        aws_access_key_id=creds['AccessKeyId'],
        aws_secret_access_key=creds['SecretAccessKey'],
        aws_session_token=creds['SessionToken']
    )

def get_credentials():
    global credentials, expiration
    sts_client = boto3.client("sts")
    utc = pytz.UTC

    # Convert current time to offset-aware datetime in UTC
    current_time_utc = datetime.now().replace(tzinfo=utc)

    # Check if credentials are None or expired
    if credentials is None or current_time_utc >= expiration - timedelta(minutes=5):
        # Assume the role
        assumed_role_object = sts_client.assume_role(
            RoleArn=f"arn:aws:iam::{AWS_ACCOUNT_ID}:role/{admin_role_name}",
            RoleSessionName="AssumeRoleSession"
        )

        # Extract temporary credentials and expiration time
        credentials = assumed_role_object['Credentials']
        expiration = credentials['Expiration']

        # Convert expiration to offset-aware datetime in UTC
        expiration = expiration.replace(tzinfo=utc)

        # Update the global boto3 client objects
        update_clients_with_credentials(credentials)


# Function to create IAM role if it doesn't exist
def create_iam_role_if_not_exists(role_name, assume_role_policy_document, managed_policies):
    existing_roles = iam.list_roles()['Roles']
    if any(role['RoleName'] == role_name for role in existing_roles):
        print(f"IAM role {role_name} already exists.")
        return

    try:
        iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(assume_role_policy_document)
        )
        print(f"IAM role {role_name} created successfully.")

        # Attach the policies
        for policy_arn in managed_policies:
            iam.attach_role_policy(
                RoleName=role_name,
                PolicyArn=policy_arn
            )
        print(f"Policies attached to IAM role {role_name} successfully.")
    except Exception as e:
        print(f"An error occurred while creating the IAM role {role_name}: {e}")


create_iam_role_if_not_exists(
    executor_role_name,
    {
        'Version': '2012-10-17',
        'Statement': [{
            'Action': 'sts:AssumeRole',
            'Effect': 'Allow',
            'Principal': {'Service': 'lambda.amazonaws.com'}
        }]
    },
    [
        'arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole',
        'arn:aws:iam::aws:policy/service-role/AWSLambdaVPCAccessExecutionRole',
    ]
)
create_iam_role_if_not_exists(
    admin_role_name,
    {
        'Version': '2012-10-17',
        'Statement': [{
            'Action': 'sts:AssumeRole',
            'Effect': 'Allow',
            'Principal': {'Service': 'lambda.amazonaws.com'}
        },{
            'Action': 'sts:AssumeRole',
            'Effect': 'Allow',
            'Principal': {'AWS': f'arn:aws:iam::{AWS_ACCOUNT_ID}:root'}
        }]
    },
    [
        'arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole',
        'arn:aws:iam::aws:policy/AmazonS3FullAccess',
        'arn:aws:iam::aws:policy/AmazonDynamoDBFullAccess',
        'arn:aws:iam::aws:policy/AmazonSNSFullAccess',
    ]
)

def create_and_attach_policy_if_needed(policy_name, policy_document, role_name):
    # List existing policies
    existing_policies = iam.list_policies(Scope='Local')['Policies']

    # Check if the policy already exists
    policy_arn = None
    for policy in existing_policies:
        if policy['PolicyName'] == policy_name:
            policy_arn = policy['Arn']
            print(f"Policy {policy_name} already exists.")
            break

    if policy_arn is None:
        try:
            # Create the policy
            create_policy_response = iam.create_policy(
                PolicyName=policy_name,
                PolicyDocument=json.dumps(policy_document)
            )
            policy_arn = create_policy_response['Policy']['Arn']
            print(f"Policy {policy_name} created successfully.")
        except Exception as e:
            print(f"An error occurred while creating the policy {policy_name}: {e}")

    # Check if the policy is already attached to the role
    attached_policies = iam.list_attached_role_policies(RoleName=role_name)['AttachedPolicies']
    if any(attached_policy['PolicyName'] == policy_name for attached_policy in attached_policies):
        print(f"Policy {policy_name} is already attached to role {role_name}.")
        return

    # Attach the policy to the role
    try:
        iam.attach_role_policy(
            RoleName=role_name,
            PolicyArn=policy_arn
        )
        print(f"Policy {policy_name} attached to role {role_name} successfully.")
    except Exception as e:
        print(f"An error occurred while attaching the policy {policy_name} to role {role_name}: {e}")

# Define the permissions policy
admin_permissions_policy = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "lambda:CreateFunction",
                "lambda:InvokeFunction",
                "lambda:UpdateFunctionConfiguration",
                "lambda:TagResource",
                "lambda:PutFunctionEventInvokeConfig",
                "elasticfilesystem:CreateAccessPoint",
                "elasticfilesystem:DeleteAccessPoint",
                "elasticfilesystem:DescribeAccessPoints",
                "elasticfilesystem:DescribeMountTargets",
                "iam:PassRole",
                "lambda:GetFunction"
            ],
            "Resource": "*"
        }
    ]
}
executor_permissions_policy = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "lambda:InvokeFunction",
                "elasticfilesystem:ClientMount",
                "elasticfilesystem:ClientWrite",
                "elasticfilesystem:ClientRead",
                "elasticfilesystem:ClientRootAccess",
                "ec2:CreateNetworkInterface",
                "ec2:DescribeNetworkInterfaces",
                "ec2:DeleteNetworkInterface"
            ],
            "Resource": "*"
        }
    ]
}
create_and_attach_policy_if_needed('LambdaExecutorPolicy', executor_permissions_policy, executor_role_name)
create_and_attach_policy_if_needed('LambdaAdminPolicy', admin_permissions_policy, admin_role_name)


def create_vpc_if_not_exists():

    # Check if VPC exists with the desired CIDR block
    vpcs = list(ec2.vpcs.filter(Filters=[{'Name': 'cidr', 'Values': ['10.45.64.0/18']}]))

    if vpcs:
        print(f"VPC already exists with CIDR block 10.45.64.0/18: {vpcs[0].id}")
        return vpcs[0].id

    # Create VPC
    vpc = ec2.create_vpc(CidrBlock='10.45.64.0/18')
    vpc.wait_until_available()
    print(f"Created VPC with ID: {vpc.id}")

    # Create a subnet
    subnet = vpc.create_subnet(CidrBlock='10.45.64.0/24')
    print(f"Created subnet with ID: {subnet.id}")

    # Create a security group
    security_group = ec2.create_security_group(
        GroupName='EFS-SG',
        Description='Security Group for EFS',
        VpcId=vpc.id
    )
    print(f"Created security group with ID: {security_group.id}")

    return vpc.id

def create_or_get_mount_targets(file_system_id):
    # Initialize EFS and EC2 clients
    global subnet_ids, security_group_ids
    ec2_client = boto3.client('ec2')
    
    # List existing mount targets
    response = efs_client.describe_mount_targets(FileSystemId=file_system_id)
    existing_mount_targets = response['MountTargets']
    
    # Fetch the VPC ID from the first mount target (assuming all are in the same VPC)
    vpc_id = existing_mount_targets[0]['VpcId'] if existing_mount_targets else None
    
    # If VPC ID is not found
    if not vpc_id:
        vpc_id = create_vpc_if_not_exists()
    
    # List all subnets in the VPC
    subnets = ec2_client.describe_subnets(Filters=[{'Name': 'vpc-id', 'Values': [vpc_id]}])
    subnet_ids = [subnet['SubnetId'] for subnet in subnets['Subnets']]
    
    # List all security groups in the VPC
    security_groups = ec2_client.describe_security_groups(Filters=[{'Name': 'vpc-id', 'Values': [vpc_id]}])
    security_group_ids = [sg['GroupId'] for sg in security_groups['SecurityGroups']]
    
    # Check if mount targets already exist for the specified subnets
    existing_subnets = [mt['SubnetId'] for mt in existing_mount_targets]
    
    # Create mount targets for missing subnets
    for subnet_id in subnet_ids:
        if subnet_id not in existing_subnets:
            print(f"Creating mount target in subnet {subnet_id}")
            efs_client.create_mount_target(
                FileSystemId=file_system_id,
                SubnetId=subnet_id,
                SecurityGroups=security_group_ids
            )
        else:
            print(f"Mount target already exists in subnet {subnet_id}")

def get_or_create_file_system(name):
    max_retries = 5  # Set the maximum number of retries
    sleep_interval = 0.2  # Set the sleep interval in seconds
    response = None
    for _ in range(max_retries):
        try:
            # Describe all file systems
            response = efs_client.describe_file_systems()
            break
        except ClientError as e:
            if e.response['Error']['Code'] == 'AccessDeniedException':
                print(f"Access denied. Retrying in {sleep_interval} seconds...")
                time.sleep(sleep_interval)
            else:
                # If the exception is not 'AccessDeniedException', re-raise it
                raise

    # Check if the file system with the given name already exists
    for file_system in response['FileSystems']:
        if file_system['Name'] == name:
            print(f"File system {name} already exists with ID: {file_system['FileSystemId']}")
            create_or_get_mount_targets(file_system['FileSystemId'])
            return file_system['FileSystemId']
    
    # If it doesn't exist, create a new file system
    print(f"Creating new file system with name: {name}")
    response = efs_client.create_file_system(
        CreationToken=name,  # Using name as a unique token
        PerformanceMode='generalPurpose',
        ThroughputMode='bursting',
        Tags=[
            {
                'Key': 'Name',
                'Value': name
            },
        ]
    )
    
    file_system_id = response['FileSystemId']
    create_or_get_mount_targets(file_system_id)
    print(f"Created new file system with ID: {file_system_id}")
    return file_system_id

# Usage
file_system_id = get_or_create_file_system('SuperDappFS')

# Function to create DynamoDB table if it doesn't exist
def create_table_if_not_exists(table_name, key_schema, attribute_definitions, read_capacity=5, write_capacity=5):
    dynamodb_client = boto3.client('dynamodb')
    existing_tables = dynamodb_client.list_tables()['TableNames']
    if table_name not in existing_tables:
        try:
            table = dynamodb.create_table(
                TableName=table_name,
                KeySchema=key_schema,
                AttributeDefinitions=attribute_definitions,
                ProvisionedThroughput={
                    'ReadCapacityUnits': read_capacity,
                    'WriteCapacityUnits': write_capacity
                }
            )
            table.wait_until_exists()  # Wait until the table is created
            print(f"Table {table_name} created successfully.")
        except Exception as e:
            print(f"An error occurred while creating the table {table_name}: {e}")
    else:
        print(f"Table {table_name} already exists.")

# Create the Users table
create_table_if_not_exists(
    'UserMetrics',
    [{'AttributeName': 'UserId', 'KeyType': 'HASH'}],
    [{'AttributeName': 'UserId', 'AttributeType': 'S'}]
)

# Create the Lambdas table
create_table_if_not_exists(
    'Lambdas',
    [
        {'AttributeName': 'UserId', 'KeyType': 'HASH'},
        {'AttributeName': 'S3Key', 'KeyType': 'RANGE'}
    ],
    [
        {'AttributeName': 'UserId', 'AttributeType': 'S'},
        {'AttributeName': 'S3Key', 'AttributeType': 'S'}
    ]
)

# Create the AccessPoints table
create_table_if_not_exists(
    'AccessPoints',
    [{'AttributeName': 'UserId', 'KeyType': 'HASH'}],
    [{'AttributeName': 'UserId', 'AttributeType': 'S'}]
)

def get_dynamodb_table(table_name):
    table = dynamodb.Table(table_name)
    return table

def wait_for_access_point_to_be_available(efs_client, access_point_id, max_retries=5, sleep_interval=0.2):
    for _ in range(max_retries):
        response = efs_client.describe_access_points(
            AccessPointId=access_point_id
        )
        if response['AccessPoints'][0]['LifeCycleState'] == 'available':
            return True
        time.sleep(sleep_interval)
    print("Max retries reached. Access point is still not available.")
    return False

def setup_efs_and_lambda(file_system_id, user_id, function_arn):
    # Function to check if an access point already exists in db
    def check_access_point_exists(user_id):
        response = ap_table.get_item(Key={'UserId': user_id})
        if 'Item' in response:
            return response['Item']
        return None

    # Check if the access point already exists
    existing_access_point_id = check_access_point_exists(user_id)

    # Create the access point if it doesn't exist
    if existing_access_point_id is None:
        efs_response = efs_client.create_access_point(
            FileSystemId=file_system_id,
            PosixUser={
                'Uid': 1001,
                'Gid': 1001
            },
            RootDirectory={
                'Path': f'/{user_id}',
                'CreationInfo': {
                    'OwnerUid': 1001,
                    'OwnerGid': 1001,
                    'Permissions': '755'
                }
            }
        )
        access_point_arn = efs_response['AccessPointArn']
        existing_access_point_id = efs_response['AccessPointId']
        # Update the Lambda function configuration to mount the EFS Access Point
        # Wait for the access point to become available
        if wait_for_access_point_to_be_available(efs_client, existing_access_point_id):
            # Now update the Lambda function configuration
            lambda_client.update_function_configuration(
                FunctionName=function_arn,
                FileSystemConfigs=[
                    {
                        'Arn': access_point_arn,
                        'LocalMountPath': '/mnt/efs'
                    },
                ],
                VpcConfig={
                    'SubnetIds': subnet_ids,
                    'SecurityGroupIds': security_group_ids,
                }
            )
        else:
            print("Failed to update function configuration. Access point is not available.")
        ap_table.put_item(
            Item={
                'UserId': user_id,
                'AccessPointId': existing_access_point_id,
            }
        )

def setup_efs_and_lambda_with_retry(file_system_id, user_id, function_arn, retries=5, delay=0.2):
    for i in range(retries):
        try:
            setup_efs_and_lambda(file_system_id, user_id, function_arn)
            return
        except Exception as e:
            if "Function not found" in str(e):
                time.sleep(delay)  # Wait for a few seconds before retrying
            else:
                raise e  # If it's another exception, raise it immediately
    raise Exception("Failed to setup EFS and Lambda after {} retries".format(retries))

# Usage
user_metrics_table = get_dynamodb_table('UserMetrics')
lambda_table = get_dynamodb_table('Lambdas')
ap_table = get_dynamodb_table('AccessPoints')

def validate_role(role_arn):
    role_name = role_arn.split('/')[-1]
    
    # Check trust relationship
    trust_policy = iam.get_role(RoleName=role_name)['Role']['AssumeRolePolicyDocument']
    has_lambda_service = any(
        stmt['Action'] == 'sts:AssumeRole' and
        stmt['Effect'] == 'Allow' and
        stmt.get('Principal', {}).get('Service') == 'lambda.amazonaws.com'
        for stmt in trust_policy.get('Statement', [])
    )
    
    if not has_lambda_service:
        raise ValueError("The role does not have the correct trust relationship.")
    
    # Check attached policies
    attached_policies = iam.list_attached_role_policies(RoleName=role_name)['AttachedPolicies']
    has_basic_execution = any(
        policy['PolicyArn'] == 'arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole'
        for policy in attached_policies
    )
    
    if not has_basic_execution:
        raise ValueError("The role does not have the AWSLambdaBasicExecutionRole policy attached.")

get_credentials()

# Execute Lambda function
# Runtime:
# Python: python3.8, python3.7, python3.6, python2.7
# Node.js: nodejs14.x, nodejs12.x, nodejs10.x
# Java: java11, java8.al2, java8
# .NET: dotnetcore3.1, dotnetcore2.1
# Go: go1.x
# Ruby: ruby2.7, ruby2.5
# Custom runtimes

# Handler:
# Python: filename.methodname (e.g., my_handler.my_function)
# Node.js: filename.methodname (e.g., index.handler)
# Java: package.Class::method (e.g., com.example.Handler::handleRequest)
# Go: filename (The Go executable itself is the handler)
# Ruby: filename.methodname (e.g., lambda_function.handler)
# .NET Core: Assembly::Namespace.ClassName::Method (e.g., Assembly::ExampleNamespace.ExampleClass::ExampleMethod)
# aws_role_arn: f'arn:aws:iam::{AWS_ACCOUNT_ID}:role/{executor_role_name} this is where you setup your resource access at runtime
class CreateLambda(BaseModel):
    # user_id is the one who is paying SUPR, we need someone responsible for paying otherwise if we use conversation_id (groups) then you 
    # can end up in a situation where some people in the group are using more resources than others and not contributing to the cost.
    # For that situation users can fork from the group by taking the creators context (AI + Cloud context), copying it working on his own
    # and collaborating with tools like Git for coding to merge changes back to official group version.
    user_id: str
    conversation_id: str
    handler: str
    s3_bucket: str
    s3_key: str
    runtime: str = None  # Make it optional
    memory_size: int = None # Make it optional
    aws_role_arn: str = None  # Make it optional

@router.post('/create_lambda')
async def create_lambda(createLambda: CreateLambda):
    try:
        get_credentials()
        # If a custom role is provided, validate it
        if createLambda.aws_role_arn:
            validate_role(createLambda.aws_role_arn)
        
        # Determine the role
        aws_role_arn = createLambda.aws_role_arn if createLambda.aws_role_arn else f'arn:aws:iam::{AWS_ACCOUNT_ID}:role/{executor_role_name}'
        
        # Determine the runtime
        runtime = createLambda.runtime if createLambda.runtime else 'python3.8'

         # Determine the memory
        memory_size = createLambda.memory_size if createLambda.memory_size else 128

        # Create the Lambda function
        response_create = lambda_client.create_function(
            FunctionName=createLambda.s3_key,
            Runtime=runtime,
            Handler=createLambda.handler,
            MemorySize=memory_size,
            Role=aws_role_arn,
            Code={
                'S3Bucket': createLambda.s3_bucket,
                'S3Key': f'{createLambda.s3_key}.zip'
            },
        )

        # Extract the ARN from the response
        function_arn = response_create['FunctionArn']
        
        # Setup EFS and Lambda for the user
        setup_efs_and_lambda_with_retry(file_system_id, createLambda.user_id, function_arn)
        
        # Store function metadata in DynamoDB
        lambda_table.put_item(
            Item={
                'UserId': createLambda.user_id,
                'S3Key': createLambda.s3_key,
                'ConversationId': createLambda.conversation_id,
                'MemorySize': memory_size,
            }
        )
        
        return {"response": "success"}
    
    except Exception as e:
        return {"error": str(e)}

class RunLambda(BaseModel):
    user_id: str
    conversation_id: str
    s3_key: str
    payload: dict = None

async def invoke_lambda_async(s3_key, payload):
    session = aioboto3.Session()
    async with session.client('lambda', aws_access_key_id=credentials['AccessKeyId'], aws_secret_access_key=credentials['SecretAccessKey'], aws_session_token=credentials['SessionToken']) as lambda_client:
        lambda_response = await lambda_client.invoke(
            FunctionName=s3_key,
            InvocationType='RequestResponse',
            Payload=payload
        )
        return lambda_response

@router.post('/run_lambda')
async def run_lambda(compute: RunLambda):
    try:
        get_credentials()
        query_args = {
            'KeyConditionExpression': Key('UserId').eq(compute.user_id) & Key('S3Key').eq(compute.s3_key)
        }
        response = lambda_table.query(**query_args)
        response = response['Items'] if 'Items' in response else None
        if response is None or len(response) == 0:
            return {"error": "Lambda not found, did you create it with createLambda?"}
        
        memory_size = response[0]['MemorySize']
        approximateInvokeCount = 1
        
        payload = json.dumps(compute.payload).encode('utf-8')
        
        start_time = time.time_ns() // 1000000
        lambda_response = await invoke_lambda_async(compute.s3_key, payload)
        
        if lambda_response is None:
            return {"error": "Lambda not executed properly"}
        if lambda_response['StatusCode'] == 200:
            approximateInvokeCount += lambda_response['ResponseMetadata']['RetryAttempts']
            end_time = time.time_ns() // 1000000
            elapsed_time = end_time - start_time
            user_metrics_table.update_item(
                Key={
                    'UserId': compute.user_id,
                },
                UpdateExpression='ADD InvocationCount :inc, TotalTimeSpentMS :time, TotalMemorySpentMB :memory',
                ExpressionAttributeValues={
                    ':inc': approximateInvokeCount,
                    ':time': elapsed_time,
                    ':memory': approximateInvokeCount * memory_size,
                },
                ReturnValues='UPDATED_NEW'
            )
            payload = json.loads(await lambda_response['Payload'].read())
            return {"response": payload}
        else:
            return {"response": lambda_response}
    except Exception as e:
        return {"error": str(e)}

class StatusUser(BaseModel):
    user_id: str

@router.post('/get_user_metrics')
async def get_user_metrics(status: StatusUser):
    try:
        get_credentials()
        response = user_metrics_table.get_item(Key={'UserId': status.user_id})
        response = response['Item'] if 'Item' in response else "Not found"
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}

@router.post('/clear_user_metrics')
async def clear_user_metrics(status: StatusUser):
    try:
        get_credentials()
        user_metrics_table.update_item(
            Key={
                'UserId': status.user_id,
            },
            UpdateExpression='SET InvocationCount = :zero, TotalTimeSpentMS = :zero, TotalMemorySpentMB = :zero',
            ExpressionAttributeValues={
                ':zero': 0
            },
            ReturnValues='UPDATED_NEW'
        )
        return {"response": "success"}
    except Exception as e:
        return {"error": str(e)}

class DeleteLambdas(BaseModel):
    user_id: str = None
    s3_key: str = None

# delete all lambdas by user or user+function
@router.post('/delete_lambdas')
async def delete_lambda(delete: DeleteLambdas):
    try:
        get_credentials()
        if delete.user_id is None and delete.s3_key is None:
            return {"error": "Either user_id or s3_key must be provided."}

        if delete.s3_key:
            if delete.user_id is None:
                return {"error": "user_id must be provided when s3_key is specified."}

            last_evaluated_key = None
            while True:
                query_args = {
                    'KeyConditionExpression': Key('UserId').eq(delete.user_id) & Key('S3Key').eq(delete.s3_key)
                }
                if last_evaluated_key:
                    query_args['ExclusiveStartKey'] = last_evaluated_key

                response = lambda_table.query(**query_args)

                for item in response['Items']:
                    lambda_client.delete_function(FunctionName=item['S3Key'])
                    lambda_table.delete_item(Key={'UserId': item['UserId'], 'S3Key': item['S3Key']})
                    time.sleep(0.05)  # Simple rate-limiting

                last_evaluated_key = response.get('LastEvaluatedKey')
                if not last_evaluated_key:
                    break

        elif delete.user_id:
            last_evaluated_key = None
            while True:
                query_args = {
                    'KeyConditionExpression': Key('UserId').eq(delete.user_id)
                }
                if last_evaluated_key:
                    query_args['ExclusiveStartKey'] = last_evaluated_key

                response = lambda_table.query(**query_args)

                for item in response['Items']:
                    lambda_client.delete_function(FunctionName=item['S3Key'])
                    lambda_table.delete_item(Key={'UserId': item['UserId'], 'S3Key': item['S3Key']})
                    time.sleep(0.05)  # Simple rate-limiting

                last_evaluated_key = response.get('LastEvaluatedKey')
                if not last_evaluated_key:
                    break
                
            user_metrics_table.delete_item(Key={'UserId': delete.user_id})

            last_evaluated_key = None
            while True:
                query_args = {
                    'KeyConditionExpression': Key('UserId').eq(delete.user_id)
                }
                if last_evaluated_key:
                    query_args['ExclusiveStartKey'] = last_evaluated_key

                response = ap_table.query(**query_args)

                for item in response['Items']:
                    efs_client.delete_access_point(AccessPointId=item['AccessPointId'])
                    ap_table.delete_item(Key={'UserId': item['UserId']})
                    time.sleep(0.05)  # Simple rate-limiting

                last_evaluated_key = response.get('LastEvaluatedKey')
                if not last_evaluated_key:
                    break
        return {"response": "Successfully deleted resources."}
    except Exception as e:
        return {"error": str(e)}

class FindLambda(BaseModel):
    user_id: str
    s3_key: str = None  # Make it optional

@router.post('/find_lambdas')
async def find_lambdas(finder: FindLambda):
    try:
        get_credentials()
        query_params = {
            'KeyConditionExpression': Key('UserId').eq(finder.user_id)
        }

        if finder.s3_key:
            query_params['KeyConditionExpression'] = query_params['KeyConditionExpression'] & Key('S3Key').eq(finder.s3_key)

        response = lambda_table.query(**query_params)
        response = response['Items'] if 'Items' in response else "Not found"
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}