import asyncio
import aiohttp
import time

async def fetch(session, url, query):
    start_time = time.time()
    async with session.post(url, data={"query": query}) as response:
        end_time = time.time()
        resp_time = end_time - start_time
        return await response.text(), resp_time

async def main():
    url = "http://localhost:8000/query_curie/"
    user_messages = [
    "How does weather forecasting work?",
    "Can you recommend a good book to read?",
    "What are some healthy eating tips?",
    "What is the best way to learn a new language?",
    "How do I start investing in the stock market?",
    "What are some effective time management techniques?",
    "Can you suggest some fun outdoor activities?",
    "How can I improve my photography skills?",
    "What are the benefits of regular exercise?",
    "How do I start a blog?",
    "Can you provide tips for job interviews?",
    "What are some popular tourist destinations?",
    "How can I improve my credit score?",
    "What are some home organization hacks?",
    "Can you recommend a good workout routine?",
    "How do I stay motivated to achieve my goals?",
    "What are some effective study techniques?",
    "Can you suggest ways to reduce stress?",
    "How do I create a budget and manage my finances?",
    "What are some strategies for effective communication?"
    ]
    
    tasks = []
    async with aiohttp.ClientSession() as session:
        for _ in range(len(user_messages)):
        # for _ in range(10):
            query = user_messages[_]
            tasks.append(fetch(session, url, query))        

        responses = await asyncio.gather(*tasks)
        # Here, responses is a list of (response, response_time) pairs for each request
        
        for response, resp_time in responses:
            print(f"Response: {response}, Response Time: {resp_time} seconds")

# Run the test
loop = asyncio.get_event_loop()
loop.run_until_complete(main())
