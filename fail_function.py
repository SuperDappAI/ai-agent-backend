import json
import boto3
import time

dynamodb = boto3.resource('dynamodb')
invocation_table = dynamodb.Table('InvocationTable')
user_metrics_table = dynamodb.Table('UserMetrics')

def lambda_handler_success(event, context):
    request_id = event['requestContext']['requestId']
    approximateInvokeCount = event['requestContext'].get('approximateInvokeCount', 1)
    payload = event.get('responsePayload', {})
    request_payload = event.get('requestPayload', {})
    user_id = request_payload.get('user_id')
    start_time = request_payload.get('start_time')
    conversation_id = request_payload.get('conversation_id')
    memory_limit_in_mb = int(context.memory_limit_in_mb)
    current_time = time.time_ns() // 1000000
    elapsed_time = current_time - start_time
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
        UpdateExpression='ADD InvocationCount :inc, TotalTimeSpentMS :time, TotalMemorySpentMB :memory',
        ExpressionAttributeValues={
            ':inc': approximateInvokeCount,
            ':time': elapsed_time,
            ':memory': approximateInvokeCount * memory_limit_in_mb,
        },
        ReturnValues='UPDATED_NEW'
    )
