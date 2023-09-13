import os


from dotenv import load_dotenv
import boto3
from boto3.dynamodb.conditions import Key
import json
import zipfile
import io
import time
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

# Load environment variables
load_dotenv()
AWS_ACCOUNT_ID = os.getenv("AWS_ACCOUNT_ID")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")
print(f"AWS_ACCOUNT_ID {AWS_ACCOUNT_ID}")
# Initialize AWS SDK
lambda_client = boto3.client('lambda')
dynamodb = boto3.resource('dynamodb')
iam = boto3.client('iam')

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
    [
        {'AttributeName': 'UserId', 'KeyType': 'HASH'},
        {'AttributeName': 'InvocationId', 'KeyType': 'RANGE'}
    ],
    [
        {'AttributeName': 'UserId', 'AttributeType': 'S'},
        {'AttributeName': 'InvocationId', 'AttributeType': 'S'},
    ]
)

# Create the Users table
create_table_if_not_exists(
    'Users',
    [{'AttributeName': 'UserId', 'KeyType': 'HASH'}],
    [{'AttributeName': 'UserId', 'AttributeType': 'S'}]
)

# Create the Lambdas table
create_table_if_not_exists(
    'Lambdas',
    [
        {'AttributeName': 'UserId', 'KeyType': 'HASH'},
        {'AttributeName': 'ConversationId', 'KeyType': 'RANGE'},
    ],
    [
        {'AttributeName': 'UserId', 'AttributeType': 'S'},
        {'AttributeName': 'ConversationId', 'AttributeType': 'S'},
    ]
)

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


executor_role_name = 'executor-role'
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
admin_role_name = 'admin-role'
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
                "iam:PassRole"
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
            ],
            "Resource": "*"
        }
    ]
}
create_and_attach_policy_if_needed('LambdaExecutorPolicy', executor_permissions_policy, executor_role_name)
create_and_attach_policy_if_needed('LambdaAdminPolicy', admin_permissions_policy, admin_role_name)
    
success_code = '''
import json
import boto3

dynamodb = boto3.resource('dynamodb')
invocation_table = dynamodb.Table('InvocationTable')

def lambda_handler_success(event, context):
    request_id = event['requestContext']['requestId']
    approximateInvokeCount = event['requestContext']['approximateInvokeCount']
    payload = event.get('responsePayload', {})
    conversation_id = event.get('conversation_id')
    user_id = event.get('user_id')
    end_time = time.perf_counter()
    elapsed_time_ms = (end_time - event.get('start_time')) * 1000

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
            ':time': elapsed_time_ms,
            ':memory': approximateInvokeCount*context.memory_limit_in_mb,    
        },
        ReturnValues='UPDATED_NEW'
    )
'''

fail_code = '''
import json
import boto3

dynamodb = boto3.resource('dynamodb')
invocation_table = dynamodb.Table('FailureTable')

def lambda_handler_fail(event, context):
    request_id = event['requestContext']['requestId']
    approximateInvokeCount = event['requestContext']['approximateInvokeCount']
    payload = event.get('responsePayload', {})
    conversation_id = event.get('conversation_id')
    user_id = event.get('user_id')
    end_time = time.perf_counter()
    elapsed_time_ms = (end_time - event.get('start_time')) * 1000

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
            ':time': elapsed_time_ms,
            ':memory': approximateInvokeCount*context.memory_limit_in_mb,    
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
    'SuccessFunction',
    'python3.8',
    f'arn:aws:iam::{AWS_ACCOUNT_ID}:role/{admin_role_name}',
    'lambda_handler_success',
    success_zip
)

# Create FailFunction if it doesn't exist
create_lambda_function_if_not_exists(
    'FailFunction',
    'python3.8',
    f'arn:aws:iam::{AWS_ACCOUNT_ID}:role/{admin_role_name}',
    'lambda_handler_fail',
    fail_zip
)

def get_dynamodb_table(table_name):
    table = dynamodb.Table(table_name)
    return table

# Usage
invocation_table = get_dynamodb_table('InvocationTable')
user_metrics_table = get_dynamodb_table('Users')
lambda_table = get_dynamodb_table('Lambdas')

def get_function_arn(function_name):
    response = lambda_client.get_function(FunctionName=function_name)
    return response['Configuration']['FunctionArn']

# Usage
success_arn = get_function_arn('SuccessFunction')
fail_arn = get_function_arn('FailFunction')

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
    user_id: str
    conversation_id: str
    runtime: str = None  # Make it optional
    handler: str
    s3_bucket: str
    s3_key: str
    memory_size: int
    role: str = None  # Make it optional

@router.post('/create_lambda')
async def create_lambda(createLambda: CreateLambda):
    createLambda.start_time = time.perf_counter()
    # pass in custom role or use default basic execution role
    role = createLambda.role
    if role is None:
        role = f'arn:aws:iam::{AWS_ACCOUNT_ID}:role/{executor_role_name}'
    runtime = createLambda.runtime
    if runtime is None:
        runtime = 'python3.8'
    response_create = lambda_client.create_function(
        FunctionName=createLambda.s3_key,
        Runtime=runtime,
        Handler=createLambda.handler,
        MemorySize=createLambda.memory_size,
        Role=role,
        Tags={
            createLambda.dict()
        },
        Code={
            'S3Bucket': createLambda.s3_bucket,
            'S3Key': f'{createLambda.s3_key}.zip'
        }
    )
    # Extract the ARN from the response
    function_arn = response['FunctionArn']

    response = lambda_client.put_function_event_invoke_config(
        FunctionName=function_arn,
        DestinationConfig={
            'OnSuccess': {
                'Destination': success_arn
            },
            'OnFailure': {
                'Destination': fail_arn
            }
        }
    )
    lambda_table.put_item(
        Item={
            'UserId': createLambda.user_id,
            'FunctionId': createLambda.s3_key,
            'ConversationId': createLambda.conversation_id,
        }
    )
    return {"response": response_create}

class RunLambda(BaseModel):
    function_name: str
    payload: dict

@router.post('/run_lambda')
def run_lambda(compute: RunLambda):
    lambda_response = lambda_client.invoke(
        FunctionName=compute.function_name,
        InvocationType='Event',
        Payload=json.dumps(compute.payload).encode('utf-8') 
    )
    return {"response": lambda_response}

class StatusLambda(BaseModel):
    user_id: str

@router.post('/get_lambda_report')
def get_lambda_status(status: StatusLambda):
    response = invocation_table.get_item(Key={'UserId': status.user_id})
    return {"response": response['Item']}

class StatusUser(BaseModel):
    user_id: str

@router.post('/get_user_metrics')
def get_user_metrics(status: StatusUser):
    response = user_metrics_table.get_item(Key={'UserId': status.user_id})
    return {"response": response['Item']}

@router.post('/clear_user_metrics')
def clear_user_metrics(status: StatusUser):
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

class FindLambda(BaseModel):
    user_id: str
    conversation_id: str = None  # Make it optional

@router.post('/find_lambdas')
def find_lambdas(finder: FindLambda):
    query_params = {
        'KeyConditionExpression': Key('UserId').eq(finder.user_id)
    }

    if finder.conversation_id:
        query_params['KeyConditionExpression'] = query_params['KeyConditionExpression'] & Key('ConversationId').eq(finder.conversation_id)

    response = lambda_table.query(**query_params)
    return {"response": response['Items']}