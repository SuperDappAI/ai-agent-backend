import os


from dotenv import load_dotenv
import boto3
from boto3.dynamodb.conditions import Key
import json
import zipfile
import io
import time
import pytz
from fastapi import APIRouter
from pydantic import BaseModel
from datetime import datetime, timedelta

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
sts_client = boto3.client("sts")

credentials = None
expiration = None
def update_clients_with_credentials(creds):
    global lambda_client, dynamodb, efs_client
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

def get_credentials():
    global credentials, expiration

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
        'arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole'
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
                "iam:PassRole",
                "elasticfilesystem:DescribeFileSystems",
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
                "elasticfilesystem:ClientRootAccess"
            ],
            "Resource": "*"
        }
    ]
}
create_and_attach_policy_if_needed('LambdaExecutorPolicy', executor_permissions_policy, executor_role_name)
create_and_attach_policy_if_needed('LambdaAdminPolicy', admin_permissions_policy, admin_role_name)

get_credentials()
def get_or_create_file_system(name):
    # Describe all file systems
    response = efs_client.describe_file_systems()
    
    # Check if the file system with the given name already exists
    for file_system in response['FileSystems']:
        if file_system['Name'] == name:
            print(f"File system {name} already exists with ID: {file_system['FileSystemId']}")
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

# Create the InvocationTable
create_table_if_not_exists(
    'InvocationTable',
    [{'AttributeName': 'UserId', 'KeyType': 'HASH'}],
    [{'AttributeName': 'UserId', 'AttributeType': 'S'}]
)

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
        {'AttributeName': 'ConversationId', 'KeyType': 'RANGE'}
    ],
    [
        {'AttributeName': 'UserId', 'AttributeType': 'S'},
        {'AttributeName': 'ConversationId', 'AttributeType': 'S'}
    ]
)

# Create the AccessPoints table
create_table_if_not_exists(
    'AccessPoints',
    [{'AttributeName': 'UserId', 'KeyType': 'HASH'}],
    [{'AttributeName': 'UserId', 'AttributeType': 'S'}]
)

success_code = '''
import json
import boto3

dynamodb = boto3.resource('dynamodb')
invocation_table = dynamodb.Table('InvocationTable')
user_metrics_table = dynamodb.Table('UserMetrics')

def lambda_handler_success(event, context):
    request_id = event['requestContext']['requestId']
    approximateInvokeCount = event['requestContext'].get('approximateInvokeCount', 1)  # Default to 1 if not provided
    payload = event.get('responsePayload', {})
    conversation_id = event.get('conversation_id')
    user_id = event.get('user_id')
    
    remaining_time = context.get_remaining_time_in_millis()
    elapsed_time = 900000 - remaining_time  # 15 minutes = 900,000 milliseconds

    # Write to DynamoDB
    invocation_table.put_item(
        Item={
            'UserId': user_id,
            'InvocationId': request_id,
            'ConversationId': conversation_id,
            'Payload': json.dumps(payload),
            'Status': 'Success'
        }
    )
    user_metrics_table.update_item(
        Key={
            'UserId': user_id,
        },
        UpdateExpression='ADD InvocationCount :inc, TotalTimeSpent :time, TotalMemorySpent :memory',
        ExpressionAttributeValues={
            ':inc': approximateInvokeCount,
            ':time': elapsed_time,
            ':memory': approximateInvokeCount * context.memory_limit_in_mb,
        },
        ReturnValues='UPDATED_NEW'
    )
'''

fail_code = '''
import json
import boto3

dynamodb = boto3.resource('dynamodb')
invocation_table = dynamodb.Table('InvocationTable')
user_metrics_table = dynamodb.Table('UserMetrics')

def lambda_handler_failure(event, context):
    request_id = event['requestContext']['requestId']
    approximateInvokeCount = event['requestContext'].get('approximateInvokeCount', 1)  # Default to 1 if not provided
    payload = event.get('responsePayload', {})
    conversation_id = event.get('conversation_id')
    user_id = event.get('user_id')
    
    remaining_time = context.get_remaining_time_in_millis()
    elapsed_time = 900000 - remaining_time  # 15 minutes = 900,000 milliseconds

    # Write to DynamoDB
    invocation_table.put_item(
        Item={
            'UserId': user_id,
            'InvocationId': request_id,
            'ConversationId': conversation_id,
            'Payload': json.dumps(payload),
            'Status': 'Failure'
        }
    )
    user_metrics_table.update_item(
        Key={
            'UserId': user_id,
        },
        UpdateExpression='ADD InvocationCount :inc, TotalTimeSpent :time, TotalMemorySpent :memory',
        ExpressionAttributeValues={
            ':inc': approximateInvokeCount,
            ':time': elapsed_time,
            ':memory': approximateInvokeCount * context.memory_limit_in_mb,
        },
        ReturnValues='UPDATED_NEW'
    )
'''
# Create ZIP file in-memory for the success function
success_buffer = io.BytesIO()
with zipfile.ZipFile(success_buffer, 'w') as z:
    z.writestr('success_function.py', success_code)
success_zip = success_buffer.getvalue()

# Create ZIP file in-memory for the fail function
fail_buffer = io.BytesIO()
with zipfile.ZipFile(fail_buffer, 'w') as z:
    z.writestr('fail_function.py', fail_code)
fail_zip = fail_buffer.getvalue()
# Function to create Lambda function if it doesn't exist
def create_lambda_function_if_not_exists(function_name, runtime, role, handler, zip_file):
    try:
        lambda_client.get_function(FunctionName=function_name)
        print(f"Lambda function {function_name} already exists.")
    except lambda_client.exceptions.ResourceNotFoundException:
        try:
            lambda_client.create_function(
                FunctionName=function_name,
                Runtime=runtime,
                Role=role,
                Handler=handler,
                Code={
                    'ZipFile': zip_file
                }
            )
            print(f"Lambda function {function_name} created successfully.")
        except Exception as e:
            print(f"An error occurred while creating the Lambda function {function_name}: {e}")

# Create SuccessFunction if it doesn't exist
create_lambda_function_if_not_exists(
    'InterpreterSuccessFunction',
    'python3.8',
    f'arn:aws:iam::{AWS_ACCOUNT_ID}:role/{admin_role_name}',
    'success_function.lambda_handler_success',
    success_zip
)

# Create FailFunction if it doesn't exist
create_lambda_function_if_not_exists(
    'InterpreterFailFunction',
    'python3.8',
    f'arn:aws:iam::{AWS_ACCOUNT_ID}:role/{admin_role_name}',
    'fail_function.lambda_handler_fail',
    fail_zip
)

def get_dynamodb_table(table_name):
    table = dynamodb.Table(table_name)
    return table

def setup_efs_and_lambda(file_system_id, user_id):
    # Function to check if an access point already exists in db
    def check_access_point_exists(user_id):
        response = ap_table.get_item(Key={'UserId': user_id})
        if 'Item' in response:
            return response['Item']
        return None

    # Check if the access point already exists
    existing_access_point_id = check_access_point_exists(file_system_id, user_id)

    # Create the access point if it doesn't exist
    if existing_access_point_id is None:
        efs_response = efs_client.create_access_point(
            FileSystemId=file_system_id,
            PosixUser={
                'Uid': 1001,
                'Gid': 1001
            },
            RootDirectory={
                'Path': user_id,
                'CreationInfo': {
                    'OwnerUid': 1001,
                    'OwnerGid': 1001,
                    'Permissions': '755'
                }
            }
        )
        access_point_arn = efs_response['AccessPointArn']
        # Update the Lambda function configuration to mount the EFS Access Point
        lambda_response = lambda_client.update_function_configuration(
            FunctionName=user_id,
            FileSystemConfigs=[
                {
                    'Arn': access_point_arn,
                    'LocalMountPath': '/mnt'
                },
            ]
        )
        ap_table.put_item(
            Item={
                'UserId': user_id,
                'AccessPointId': existing_access_point_id,
            }
        )

        print(f"EFS Access Point and Lambda function updated successfully. Access Point ARN: {access_point_arn}")


# Usage
invocation_table = get_dynamodb_table('InvocationTable')
user_metrics_table = get_dynamodb_table('UserMetrics')
lambda_table = get_dynamodb_table('Lambdas')
ap_table = get_dynamodb_table('AccessPoints')

def get_function_arn(function_name):
    response = lambda_client.get_function(FunctionName=function_name)
    return response['Configuration']['FunctionArn']

# Usage
success_arn = get_function_arn('InterpreterSuccessFunction')
fail_arn = get_function_arn('InterpreterFailFunction')

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
# role: f'arn:aws:iam::{AWS_ACCOUNT_ID}:role/{executor_role_name} this is where you setup your resource access at runtime
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
    role: str = None  # Make it optional

@router.post('/create_lambda')
async def create_lambda(createLambda: CreateLambda):
    try:
        get_credentials()
        # If a custom role is provided, validate it
        if createLambda.role:
            validate_role(createLambda.role)

        # Setup EFS and Lambda for the user
        setup_efs_and_lambda(file_system_id, createLambda.user_id)
        
        # Determine the role
        role = createLambda.role if createLambda.role else f'arn:aws:iam::{AWS_ACCOUNT_ID}:role/{executor_role_name}'
        
        # Determine the runtime
        runtime = createLambda.runtime if createLambda.runtime else 'python3.8'
        
         # Determine the memory
        memory_size = createLambda.memory_size if createLambda.memory_size else '128'
        
        
        # Create the Lambda function
        response_create = lambda_client.create_function(
            FunctionName=createLambda.s3_key,
            Runtime=runtime,
            Handler=createLambda.handler,
            MemorySize=memory_size,
            Role=role,
            Tags=createLambda.dict(),
            Code={
                'S3Bucket': createLambda.s3_bucket,
                'S3Key': f'{createLambda.s3_key}.zip'
            }
        )
        
        # Extract the ARN from the response
        function_arn = response_create['FunctionArn']
        
        # Set the function's event invoke config
        lambda_client.put_function_event_invoke_config(
            FunctionName=function_arn,
            DestinationConfig={
                'OnSuccess': {'Destination': success_arn},
                'OnFailure': {'Destination': fail_arn}
            }
        )
        
        # Store function metadata in DynamoDB
        lambda_table.put_item(
            Item={
                'UserId': createLambda.user_id,
                'FunctionId': createLambda.s3_key,
                'ConversationId': createLambda.conversation_id,
            }
        )
        
        return {"response": response_create}
    
    except Exception as e:
        return {"error": str(e)}

class RunLambda(BaseModel):
    function_name: str
    payload: dict

@router.post('/run_lambda')
def run_lambda(compute: RunLambda):
    try:
        get_credentials()
        lambda_response = lambda_client.invoke(
            FunctionName=compute.function_name,
            InvocationType='Event',
            Payload=json.dumps(compute.payload).encode('utf-8') 
        )
        return {"response": lambda_response}
    except Exception as e:
        return {"error": str(e)}

class StatusLambda(BaseModel):
    user_id: str

@router.post('/get_lambda_report')
def get_lambda_status(status: StatusLambda):
    try:
        get_credentials()
        response = invocation_table.get_item(Key={'UserId': status.user_id})
        response = response['Item'] if 'Item' in response else "Not found"
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}

class StatusUser(BaseModel):
    user_id: str

@router.post('/get_user_metrics')
def get_user_metrics(status: StatusUser):
    try:
        get_credentials()
        response = user_metrics_table.get_item(Key={'UserId': status.user_id})
        response = response['Item'] if 'Item' in response else "Not found"
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}

@router.post('/clear_user_metrics')
def clear_user_metrics(status: StatusUser):
    try:
        get_credentials()
        response = user_metrics_table.update_item(
            Key={
                'UserId': status.user_id,
            },
            UpdateExpression='SET InvocationCount = :zero, TotalTimeSpent = :zero, TotalMemorySpent = :zero',
            ExpressionAttributeValues={
                ':zero': 0
            },
            ReturnValues='UPDATED_NEW'
        )
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}

class DeleteLambdas(BaseModel):
    user_id: str = None
    conversation_id: str = None

# delete all lambdas by user or user+conversation
@router.post('/delete_lambdas')
def delete_lambda(delete: DeleteLambdas):
    try:
        get_credentials()
        if delete.user_id is None and delete.conversation_id is None:
            return {"error": "Either user_id or conversation_id must be provided."}

        if delete.conversation_id:
            if delete.user_id is None:
                return {"error": "user_id must be provided when conversation_id is specified."}

            last_evaluated_key = None
            while True:
                query_args = {
                    'KeyConditionExpression': Key('UserId').eq(delete.user_id) & Key('ConversationId').eq(delete.conversation_id)
                }
                if last_evaluated_key:
                    query_args['ExclusiveStartKey'] = last_evaluated_key

                response = lambda_table.query(**query_args)

                for item in response['Items']:
                    lambda_client.delete_function(FunctionName=item['FunctionId'])
                    lambda_table.delete_item(Key={'UserId': item['UserId'], 'ConversationId': item['ConversationId']})
                    time.sleep(0.2)  # Simple rate-limiting

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
                    lambda_client.delete_function(FunctionName=item['FunctionId'])
                    lambda_table.delete_item(Key={'UserId': item['UserId'], 'ConversationId': item['ConversationId']})
                    time.sleep(0.2)  # Simple rate-limiting

                last_evaluated_key = response.get('LastEvaluatedKey')
                if not last_evaluated_key:
                    break

            invocation_table.delete_item(Key={'UserId': delete.user_id})
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
                    time.sleep(0.2)  # Simple rate-limiting

                last_evaluated_key = response.get('LastEvaluatedKey')
                if not last_evaluated_key:
                    break
        return {"response": "Successfully deleted resources."}
    except Exception as e:
        return {"error": str(e)}

class FindLambda(BaseModel):
    user_id: str
    conversation_id: str = None  # Make it optional

@router.post('/find_lambdas')
def find_lambdas(finder: FindLambda):
    try:
        get_credentials()
        query_params = {
            'KeyConditionExpression': Key('UserId').eq(finder.user_id)
        }

        if finder.conversation_id:
            query_params['KeyConditionExpression'] = query_params['KeyConditionExpression'] & Key('ConversationId').eq(finder.conversation_id)

        response = lambda_table.query(**query_params)
        response = response['Item'] if 'Item' in response else "Not found"
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}