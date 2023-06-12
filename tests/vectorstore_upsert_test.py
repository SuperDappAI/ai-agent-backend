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
    url = "http://localhost:8000/query_async/"

    mixed_questions= [
    "Hi, can you help me with a technical issue?",
    "How does weather forecasting work?",
    "What is the square root of 64?",
    "Can you recommend a good book to read?",
    "Solve for x: 2x + 5 = 15",
    "What are some healthy eating tips?",
    "What is the best way to learn a new language?",
    "What is the value of pi (π)?",
    "I'm having trouble connecting to the internet.",
    "Evaluate the expression: 3 * (8 - 2)",
    "Can you suggest a fun outdoor activity?",
    "How can I improve my photography skills?",
    "Can you recommend a good programming tutorial?",
    "What are the benefits of regular exercise?",
    "What are the system requirements for your software?",
    "How do I start investing in the stock market?",
    "What are some effective time management techniques?",
    "Simplify the fraction: 12/36",
    "Can you explain how to use this feature?",
    "What is the area of a circle with radius 5?"
    "Can you recommend a good programming tutorial?",
    "What are the benefits of regular exercise?",
    "What are the system requirements for your software?",
    "How do I start investing in the stock market?",
    "What are some effective time management techniques?",
    "Simplify the fraction: 12/36",
    "Can you explain how to use this feature?",
    "What is the area of a circle with radius 5?"
    "What is the best way to learn a new language?",
    "What is the value of pi (π)?",
    "I'm having trouble connecting to the internet.",
    "Evaluate the expression: 3 * (8 - 2)",
    "Can you suggest a fun outdoor activity?",
    "How can I improve my photography skills?",
    "Can you recommend a good programming tutorial?",
    ]

    tasks = []
    async with aiohttp.ClientSession() as session:
        for _ in range(len(mixed_questions)):
        # for _ in range(10):
            query = mixed_questions[_]
            tasks.append(fetch(session, url, query))        

        responses = await asyncio.gather(*tasks)
        
        for response, resp_time in responses:
            print(f"Response: {response}, Response Time: {resp_time} seconds")

# Run 
loop = asyncio.get_event_loop()
loop.run_until_complete(main())

