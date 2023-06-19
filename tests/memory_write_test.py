import asyncio
import aiohttp
import time
import random

async def fetch(session, url, query):
    start_time = time.time()
    async with session.post(url, data=query) as response:
        end_time = time.time()
        resp_time = end_time - start_time
        return await response.text(), resp_time

async def main():
    url = "http://localhost:8000/push_memory/"

    users = ['mickey','johnny','superbunny55']
    chat_history = [
    {
        "user": "What is the capital of France?",
        "assistant": "The capital of France is Paris."
    },
    {
        "user": "What is the tallest mountain in the world?",
        "assistant": "The tallest mountain in the world is Mount Everest."
    },
    {
        "user": "What is the distance between the Earth and the Moon?",
        "assistant": "The average distance from the Moon to the Earth is about 238,855 miles."
    },
    {
        "user": "What's the speed of light?",
        "assistant": "The speed of light in a vacuum is approximately 299,792 kilometers per second."
    },
    {
        "user": "Who wrote 'Pride and Prejudice'?",
        "assistant": "'Pride and Prejudice' was written by Jane Austen."
    },
    {
        "user": "When was the Declaration of Independence signed?",
        "assistant": "The Declaration of Independence was signed on July 4, 1776."
    },
    {
        "user": "How many planets are there in our solar system?",
        "assistant": "There are 8 planets in our solar system."
    },
    {
        "user": "Who painted the Mona Lisa?",
        "assistant": "The Mona Lisa was painted by Leonardo da Vinci."
    },
    {
        "user": "What's the square root of 64?",
        "assistant": "The square root of 64 is 8."
    },
    {
        "user": "Who is the current president of the United States?",
        "assistant": "As of my knowledge cutoff in September 2021, the President of the United States is Joe Biden. Please verify from a current source."
    },
    {
        "user": "Who discovered gravity?",
        "assistant": "Gravity was discovered by Sir Isaac Newton."
    },
    {
        "user": "How many continents are there in the world?",
        "assistant": "There are seven continents in the world."
    },
    {
        "user": "What is the population of India?",
        "assistant": "As of my knowledge cutoff in September 2021, the estimated population of India is over 1.3 billion. Please verify from a current source."
    },
    {
        "user": "What is photosynthesis?",
        "assistant": "Photosynthesis is a process used by plants and other organisms to convert light energy, usually from the sun, into chemical energy that can be used to fuel the organisms' activities."
    },
    {
        "user": "Who invented the telephone?",
        "assistant": "The telephone was invented by Alexander Graham Bell."
    },
    {
        "user": "How many days are there in a year?",
        "assistant": "There are usually 365 days in a year. However, in a leap year, there are 366 days."
    },
    {
        "user": "When was the internet invented?",
        "assistant": "The basic ideas of the internet began to take shape in the 1960s. The network we know today as the 'internet' was first switched on in 1969."
    },
    {
        "user": "How far is Mars from Earth?",
        "assistant": "The distance from Earth to Mars varies depending on their relative positions in their orbits. It can range from about 34 million miles to over 250 million miles."
    },
    {
        "user": "Who is Albert Einstein?",
        "assistant": "Albert Einstein was a physicist who developed the theory of relativity, one of the two pillars of modern physics. His work is also known for its influence on the philosophy of science."
    },
    {
        "user": "What is the atomic number of Hydrogen?",
        "assistant": "The atomic number of Hydrogen is 1."
    },
    {
        "user": "Who won the World Series in 2020?",
        "assistant": "The Los Angeles Dodgers won the World Series in 2020."
    },
    {
        "user": "What is the formula for water?",
        "assistant": "The chemical formula for water is H2O."
    },
    {
        "user": "How long does it take for the Earth to orbit the Sun?",
        "assistant": "It takes approximately 365.25 days for the Earth to orbit the Sun."
    },
    {
        "user": "What is Pi?",
        "assistant": "Pi is a mathematical constant that is the ratio of a circle's circumference to its diameter. It is approximately equal to 3.14159."
    },
    {
        "user": "What is the largest ocean on Earth?",
        "assistant": "The Pacific Ocean is the largest ocean on Earth."
    },
    {
        "user": "What's the boiling point of water?",
        "assistant": "The boiling point of water is 100 degrees Celsius or 212 degrees Fahrenheit at standard atmospheric pressure."
    },
    {
        "user": "What is the human body temperature in Celsius?",
        "assistant": "The average normal body temperature is generally accepted as 37 degrees Celsius."
    },
    {
        "user": "What is the capital of Australia?",
        "assistant": "The capital of Australia is Canberra."
    },
    {
        "user": "Who wrote the Harry Potter series?",
        "assistant": "The Harry Potter series was written by J.K. Rowling."
    },
    {
        "user": "What is the freezing point of water?",
        "assistant": "The freezing point of water is 0 degrees Celsius or 32 degrees Fahrenheit at standard atmospheric pressure."
    }
] 

    tasks = []
    async with aiohttp.ClientSession() as session:
        for _ in range(len(chat_history)):
        # for _ in range(10):
            message = chat_history[_]['user']
            llm_response = chat_history[_]['assistant']
            user_id = random.choice(users)
            query = {
                "message": message,
                "llm_response": llm_response,
                "user_id": user_id
                }
            tasks.append(fetch(session, url, query))        

        responses = await asyncio.gather(*tasks)
        
        for response, resp_time in responses:
            print(f"Response: {response}, Response Time: {resp_time} seconds")

# Run 
loop = asyncio.get_event_loop()
loop.run_until_complete(main())

