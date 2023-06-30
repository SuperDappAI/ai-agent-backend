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

    users = ['elizabeth','joe','pierre']

    elizabeth_chat_history = [
    {
    "user": "My name is Elizabeth. I live in Seattle and I love gardening. Can you give me some tips on how to care for roses in my city's climate?",
    "assistant": "Nice to meet you, Elizabeth. Seattle's climate, known for its cool, wet winters and warm, dry summers, can be great for growing roses. To start, it's important to choose varieties that are resistant to common rose diseases, as the humidity can promote fungal growth. Second, make sure your roses are in a location with at least six hours of sunlight daily. Despite Seattle's reputation for rain, roses need well-draining soil to avoid root diseases, so consider raising your rose beds or adding organic matter to your soil to improve its structure. Finally, be aware that while your roses might not need as much watering as in other climates, they'll still need deep watering in the summer. Monitor your plants carefully, as the dry summers can stress them. Remember, each rose variety is unique, so it's best to research the specific care requirements for the types you have."
    },
    {
    "user": "That's really helpful, thank you! I also want to know the best time to prune them in Seattle?",
    "assistant": "In Seattle, the best time to prune roses is usually in late winter or early spring, around late February or March, when new growth begins to emerge. It's important to prune before the buds break into new growth, but after the threat of a hard freeze is over. This can help to prevent damage from cold weather, while encouraging the roses to start producing new growth for the upcoming season. Always use sharp and clean pruners to make clean cuts and avoid damaging your roses."
    },
    {
    "user": "Thanks, I'll make a note of that. Changing the subject, I'm thinking of visiting France next year. Do you have any advice or recommendations?",
    "assistant": "France is a fantastic destination, rich in history, art, culture, and gastronomy. It's hard to give specific advice without knowing your preferences, but here are a few general tips: 1) Paris is a must-see for first-time visitors. Don't miss the Eiffel Tower, Louvre Museum, and the charming Montmartre neighborhood. 2) Explore the French Riviera, known for its beautiful beaches and glamorous lifestyle. 3) The Loire Valley is known for its stunning châteaux and vineyards. 4) If you love history, Normandy, where D-Day occurred, can be an impactful visit. 5) For food lovers, Lyon is a gastronomical paradise. Make sure to try regional specialties wherever you go. Also, while France is generally safe, like anywhere, stay aware of your surroundings, especially in busy tourist areas where pickpocketing can be a problem."
    },
    {
    "user": "Interesting, I'm also a huge fan of wine. Could you recommend some French wines that I should definitely try while I'm there?",
    "assistant": "Certainly, France is globally renowned for its wine. Here are a few suggestions: 1) Bordeaux: Known for their red blend wines, typically a mix of Cabernet Sauvignon and Merlot. 2) Burgundy: Famous for its Pinot Noir and Chardonnay. 3) Champagne: The only region in the world that can legally produce Champagne. 4) Rhône Valley: Produces a range of red, white, and rosé wines, but it's best known for its full-bodied reds. 5) Alsace: Known for its white wines, like Riesling and Gewürztraminer. Also, don't miss the opportunity to visit a vineyard or two if you can, as the process of making wine is as fascinating as the wine itself."
    },
    {
    "user": "I enjoy painting as a hobby. Could you give me some tips on how to improve my skills?",
    "assistant": "Absolutely, Elizabeth. Here are some tips to improve your painting skills: 1) Practice Regularly: This is the most fundamental tip. The more you paint, the more your skills will improve. 2) Study the Masters: Learn from the works of great artists. Analyze their style, techniques, color mixing, and composition. 3) Attend Workshops or Classes: You can gain valuable feedback and insights from instructors and peers. 4) Understand Color Theory: Learn how to mix colors and understand the color wheel. 5) Experiment: Don’t stick to one style or medium. Try different techniques, styles, and mediums. 6) Paint From Life: This can be more challenging than painting from a photograph, but it will improve your observation skills and understanding of light. 7) Keep a Sketchbook: Sketch regularly to improve your drawing skills, which are the foundation of painting."
    },
    {
    "user": "Thank you for all these tips! One last question for now, what's the best way to store my paintings to ensure they last?",
    "assistant": "Great question, Elizabeth. Here are a few tips to properly store your paintings: 1) Climate Control: Avoid storing art in basements or attics. A climate-controlled area is best to avoid damage from humidity and temperature changes. 2) Avoid Sunlight: Direct sunlight can fade artwork over time. Store in a dark place if possible. 3) Vertical Storage: Store paintings vertically, like in a library, with sufficient padding between each. Avoid stacking them directly on top of each other. 4) Cover: Cover with a breathable material like cotton sheets. Avoid plastic as it can trap moisture. 5) Proper Framing: Use acid-free materials for framing to prevent yellowing over time. 6) Clean: Make sure the artwork is clean before storage. Dust and dirt can cause long-term damage. If you're ever in doubt, it may be best to consult with a professional art curator or restorer."
    }
    ]
    
    joe_chat_history = [
    {
    "user": "Hello there! I'm Joe. I have a passion for physics, and I'm trying to understand quantum mechanics. Can you explain the concept of superposition to me?",
    "assistant": "Hello, Joe! Sure, I'd be happy to explain. In quantum mechanics, superposition refers to the phenomenon where a quantum system can exist in multiple states or places simultaneously. The most famous example is Schrödinger's cat, where the cat in the box is both alive and dead until it's observed. This seems counter-intuitive because it's different from what we see in our day-to-day life. However, in the quantum world, particles like electrons can be in multiple places at the same time. It's important to note that this isn't about a particle being in two places due to uncertainty or lack of knowledge, but it's about the particle literally occupying two states until an observation or measurement is made. When such a measurement is made, the system 'collapses' from the superposition of states into one specific state."
    },
    {
    "user": "That's fascinating! On another note, I've been meaning to visit the Griffith Observatory. It's only an hour away from my place. Do you have any tips?",
    "assistant": "That's great, Joe! Visiting an observatory can be an enriching experience. Here are a few tips: 1) Check the weather: Ensure the skies are clear for better viewing. 2) Visit at the right time: If possible, aim to visit during a celestial event or when the moon isn't full, as moonlight can wash out faint objects. 3) Participate in guided tours: Many observatories have experts who can provide more context about the exhibits. 4) Dress Warmly: It can get cold at night, even in warmer climates. 5) Research beforehand: Learn about what you might see during your visit. This will enhance your understanding and enjoyment. The Griffith Observatory is an excellent facility with a planetarium, exhibits, and telescopes for public use. Enjoy your visit!"
    },
    {
    "user": "I'll keep those tips in mind, thank you! Now, changing the subject, I love mathematics. I'm trying to understand the concept of integrals. Could you help with that?",
    "assistant": """Certainly, Joe! In calculus, integration is one of the two main operations, the other one being differentiation. The integral of a function is a generalization of area under a curve and can be understood in several ways.

    From a physics perspective, you can consider integration as a process of adding up an infinite number of infinitesimal quantities. For instance, if you have a velocity-time graph of an object, the integral of the velocity function gives you the total distance traveled by the object over a period of time.

    From a geometric perspective, imagine a curve on a graph. The integral of the function that produces that curve, between two points, is the area under the curve between those two points.

    The symbol for an integral is ∫, and the process of finding an integral is called integrating.

    Remember, the fundamental theorem of calculus connects differentiation and integration, stating that one is the reverse process of the other. It's a complex subject, but I hope this gives you a good starting point!"""
    },
    {
    "user": "Wow, I never looked at it that way. I'll dive deeper into integrals using this as a base. You know, I've always loved science. I even set up a small lab in my basement. I'm currently trying to learn more about chemistry, can you explain what a mole is in chemistry?",
    "assistant": """Absolutely, Joe! A 'mole' is a unit of measurement used in chemistry. It's one of the fundamental concepts that connects the macroscopic world we see and interact with every day, to the microscopic world of atoms and molecules.

    One mole of any substance contains the same number of elementary entities (like atoms, molecules, or other particles) as there are atoms in exactly 12 grams of carbon-12. This number is known as Avogadro's number, and it's incredibly large, approximately 6.022 x 10^23.

    So, when a chemist measures out 'one mole' of a substance, they're getting an Avogadro's number of atoms or molecules of that substance. This concept is fundamental for understanding reactions in chemistry, as reactions happen between molecules or atoms on a one-to-one basis, or sometimes two-to-one or other ratios, but always in discrete units.

    Think of it like a 'chemist's dozen'. Just as a 'baker's dozen' means 13 of something, a 'chemist's dozen' (or a mole) means around 602 sextillion (a 6 followed by 23 zeros) of something!"""
    },
    {
    "user": "Incredible! The more I learn about science, the more fascinating it gets. Now I'm wondering about something less theoretical. Since I live close to Los Angeles, earthquakes are a common occurrence here. Could you explain how seismologists predict earthquakes?",
    "assistant": """Predicting earthquakes with precision is currently beyond our scientific capabilities due to the complex nature of tectonic activity. However, seismologists can forecast or estimate the likelihood of earthquakes occurring in a region over a certain period. This is done using historical earthquake data, understanding of plate tectonics, and geological mapping of faults. They look at patterns in where stress is building up along fault lines and where it's been released in the past.

    When an earthquake does occur, modern seismometers can detect and measure the intensity of the quake, providing data that helps refine future forecasts. Technologies like GPS and radar are also used to monitor minute changes in the Earth's crust due to tectonic activity.

    That said, predicting the exact time, location, and magnitude of an earthquake remains a major challenge in earth science. As a resident of an earthquake-prone area, it's important to have a safety plan and emergency supplies ready."""
    },
    {
    "user": "Safety first, indeed. I've prepared an earthquake safety kit at home. Speaking of safety, as I experiment in my lab, I know how crucial it is. Can you give me some general safety rules I should always follow in my home lab?",
    "assistant": """Absolutely, Joe. Safety should always be the first priority in any lab, including a home lab. Here are some general safety rules:

    Wear Proper Lab Attire: Always wear a lab coat, safety goggles, and gloves to protect your skin and eyes. Don't wear open-toed shoes in the lab.
    Know Your Chemicals: Understand the properties, potential hazards, and safe disposal methods of the chemicals you're using. Store chemicals properly and keep Material Safety Data Sheets (MSDS) for each chemical you use.
    Ventilation: Make sure your lab is well-ventilated to avoid accumulation of harmful fumes.
    No Food or Drink: Never eat or drink in a lab to prevent ingestion of harmful substances.
    Equipment Safety: Be fully aware of how to safely use your equipment. Inspect it regularly to ensure it's in good working condition.
    Emergency Preparedness: Have a fire extinguisher and first aid kit readily accessible. Also, know the location of the nearest eye wash station and safety shower.
    Cleanliness: Keep your lab space clean and organized. Dispose of waste properly and immediately clean up any spills.
    Supervision: If you're conducting a particularly risky experiment, consider having another adult present.
    Remember, safety always comes first, so never take shortcuts that compromise it."""
    }, 
    ]
    
    pierre_chat_history = [
    {
    "user": "Hello there! I'm Pierre. I'm a computer science student at the University of Washington.",
    "assistant": "Hello, Pierre! How may I assist you with your studies in computer science today?"
    },
    {
    "user": "I'm having a bit of trouble with Python. Can you help me understand how decorators work?",
    "assistant": "Sure, Pierre. In Python, decorators are a way to modify or enhance the behavior of a function or a class. They allow us to wrap another function in order to extend the behavior of the wrapped function, without permanently modifying it. A decorator in Python is implemented as a callable that returns a callable. Here's a simple example of a decorator: \n\npython\n def simple_decorator(function):\n def wrapper():\n print('This happens before the function is called.')\n function()\n print('This happens after the function is called.')\n return wrapper\n\n@simple_decorator\ndef say_hello():\n print('Hello!')\n\n\nIn this example, @simple_decorator is used to 'decorate' the function say_hello(). When say_hello() is called, it will now print the 'before' and 'after' messages as well."
    },
    {
    "user": "That's very helpful, thank you. Now I'm wondering how to handle exceptions in Python.",
    "assistant": "Handling exceptions in Python is done with a try...except block. When the code in the try block encounters an error, instead of stopping the program, it jumps to the except block and executes that code instead. Here's a simple example:\n\npython\ntry:\n x = 1 / 0 # This will cause a ZeroDivisionError\nexcept ZeroDivisionError:\n x = 0\n print('Handled a ZeroDivisionError. x is now', x)\n\n\nIn this example, the code tries to divide 1 by 0, which is impossible and raises a ZeroDivisionError. Instead of the program stopping, it executes the except block and sets x to 0, then prints a message. Exception handling is a key part of writing robust, error-tolerant Python code."
    },
    {
    "user": "Got it, that makes sense. By the way, do you have the result of the last Mariners game?",
    "assistant": "Yes, I do. The Seattle Mariners played against the Texas Rangers last night. The Mariners won the game with a score of 5-3."
    },
    {
    "user": "That's great! They're having a good season. Now back to coding, what are Generators in Python?",
    "assistant": "Generators are a type of iterable in Python, like lists or tuples. They're a special kind of function that return a lazy iterator. These are objects that you can loop over like a list. However, unlike lists, they are lazy and produce items one at a time and only when asked. This makes them much more memory efficient when dealing with large amounts of data. Here's an example:\n\npython\ndef count_up_to(n):\n i = 1\n while i <= n:\n yield i\n i += 1\n\n\nThe count_up_to function is a generator that will yield numbers from 1 up to the n you give it. You can use it in a for loop like this:\n\npython\nfor number in count_up_to(5):\n print(number)\n\n\nThis will print the numbers 1 through 5, each on their own line."
    },
    {
    "user": "I see, generators seem quite useful! What's the score of the latest Sounders FC game?",
    "assistant": "Indeed, the Seattle Sounders FC played against the LA Galaxy yesterday. The Sounders won the match with a score of 2-1."
    },
    {
    "user": "Can you explain the concept of recursion in programming?",
    "assistant": "Sure, Pierre. Recursion is a concept where a function calls itself as a subroutine. This can be particularly useful when you need to walk through 'nested' structures, which are structures that can contain instances of themselves, like trees and linked lists. However, each recursive call requires additional memory, so it can be more memory-intensive than a loop-based equivalent. Here's a classic example - a recursive implementation of the factorial function:\n\npython\ndef factorial(n):\n if n == 0:\n return 1\n else:\n return n * factorial(n-1)\n\n\nThis function calls itself with smaller and smaller values of n until it hits the 'base case' of n == 0. Then it returns 1, which gets multiplied with all the ns it 'collected' on the way down, resulting in the factorial of the original n."
    },
    {
    "user": "Interesting! I'll try to implement some recursive functions. Can you tell me about the most recent Huskies baseball game?",
    "assistant": "Absolutely, the Washington Huskies recently played a game against the Oregon Ducks. The Huskies triumphed with a score of 7-4. It was a very exciting game!"
    },
    {
    "user": "I'm happy to hear that! Now I'm curious, how does multithreading work in Python?",
    "assistant": "Multithreading is a way to run multiple threads (smallest units of execution) in parallel. In Python, you can use the threading module to create and manage threads. However, due to the Global Interpreter Lock (GIL) in Python, threads do not truly run in parallel except for some cases involving I/O operations. Here's an example of creating two threads that run in 'parallel':\n\npython\nimport threading\n\n def print_numbers():\n for i in range(10):\n print(i)\n\n def print_letters():\n for letter in 'abcdefghij':\n print(letter)\n\n t1 = threading.Thread(target=print_numbers)\n t2 = threading.Thread(target=print_letters)\n\n t1.start() # Start the execution of the first thread\n t2.start() # Start the execution of the second thread\n\n t1.join() # Wait for the first thread to finish\n t2.join() # Wait for the second thread to finish\n\n\nIn this example, the print_numbers and print_letters functions each run in their own thread."
    },
    {
    "user": "This has been really informative. What's the latest score for the Seattle Mariners?",
    "assistant": "The Seattle Mariners had a match against the Boston Red Sox today. They performed splendidly and secured a victory with a final score of 4-2."
    },
    {
    "user": "I'm trying to write a function that calculates the factorial of a number using recursion. Here's what I have but it's not working:\n\npython\n def factorial(n):\n if n = 0:\n return 1\n return n * factorial(n)\n\nCan you help?",
    "assistant": "Absolutely, Pierre! You have a couple of issues here. Firstly, in your if statement, you're using a single equals sign (=), which is assignment. You should use two equals signs (==) for comparison. Secondly, you're not decrementing n in your recursive call. Here's the corrected code:\n\npython\n def factorial(n):\n if n == 0:\n return 1\n return n * factorial(n-1)\n"
    },
    {
    "user": "Ah, I see! I'm also having some trouble with a piece of code that's supposed to find the index of a number in a list:\n\npython\n def find_index(lst, n):\n for i in lst:\n if lst[i] == n:\n return i\n\nIt doesn't seem to work as expected. Any thoughts?",
    "assistant": "Yes, the problem is that you're using i to iterate over the elements of the list, but then you're treating i as an index. When you use for i in lst:, i takes on the values of the items in the list, not their indices. You can use enumerate(lst) to get both index and value. Here's the corrected code:\n\npython\n def find_index(lst, n):\n for i, val in enumerate(lst):\n if val == n:\n return i\n"
    },
    {
    "user": "Oh, I see. Thanks for the clarification! Now, I have this code to reverse a string but it's not working:\n\npython\n def reverse_string(s):\n return s[-1:0:-1]\n\nWhat am I doing wrong here?",
    "assistant": "The issue is with your slice. In Python, the end index in a slice is exclusive, so 0 won't be included. When you're using a negative step -1 to reverse the string, your end index should be beyond the start of the string. You can simply leave it empty to represent 'all the way to the end (or start, since you're stepping backwards)'. Here's the corrected code:\n\npython\n def reverse_string(s):\n return s[::-1]\n"
    },
    {
    "user": "Thanks! Now, I'm working on a function that finds the maximum number in a list. Here's what I've got:\n\npython\n def find_max(lst):\n max_num = 0\n for num in lst:\n if num > max_num:\n max_num = num\n return max_num\n\nBut it seems to be failing for lists with all negative numbers. Can you help me figure this out?",
    "assistant": "Sure, the issue here is that you're initializing max_num as 0. If all the numbers in the list are negative, max_num will stay 0 as it's greater than any negative number. A better approach is to initialize max_num as the first element in the list. Here's the corrected code:\n\npython\n def find_max(lst):\n max_num = lst[0]\n for num in lst:\n if num > max_num:\n max_num = num\n return max_num\n"
    },
    {
    "user": "Ah, that makes sense. I have another piece of code that's supposed to count the number of occurrences of a character in a string:\n\npython\n def count_char(s, c):\n count = 0\n for char in s:\n if char == s:\n count += 1\n return count\n\nBut it doesn't seem to be giving me the correct results. What's the problem?",
    "assistant": "The issue is in your if condition. You're comparing char (the current character in the string) with s (the entire string). You should be comparing char with c, the character you're counting. Here's the corrected code:\n\npython\n def count_char(s, c):\n count = 0\n for char in s:\n if char == c:\n count += 1\n return count\n"
    },
    {
    "user": "I see my mistake now. Thank you! I also have this function to remove all occurrences of a character from a string:\n\npython\n def remove_char(s, c):\n return s.replace(c, '')\n\nBut it doesn't seem to be removing anything. What's wrong?",
    "assistant": "From the code you've posted, it seems correct for the purpose of removing all occurrences of a character from a string in Python. If it's not working, make sure you're calling the function correctly and that the character c you're passing to the function is indeed in the string s. Also remember that str.replace() is case-sensitive, so it won't replace lowercase characters with their uppercase counterparts, or vice versa."
    },
    {
    "user": "My mistake, I was using the function incorrectly. I also have this code that's supposed to create a list of squares from 1 to n:\n\npython\n def squares(n):\n return [i * i for i in n]\n\nBut I'm getting an error when I try to call it. What's going wrong?",
    "assistant": "The issue here is that you're trying to iterate over n, which is a number, not an iterable. You probably meant to iterate over a range of numbers from 1 to n. Here's the corrected code:\n\npython\n def squares(n):\n return [i * i for i in range(1, n+1)]\n"
    },
    {
    "user": "Ah, that was a silly mistake. Now, I have this function that's supposed to print a pyramid of asterisks but it's not working:\n\npython\n def print_pyramid(n):\n for i in range(n):\n print(' ' * (n - i) + '*' * (2 * i + 1))\n\nWhat's wrong with it?",
    "assistant": "Your code is almost correct, but the range of your for loop should go up to n+1 instead of n because Python's range() function doesn't include the stop value. Here's the corrected code:\n\npython\n def print_pyramid(n):\n for i in range(n+1):\n print(' ' * (n - i) + '*' * (2 * i + 1))\n"
    },
    {
    "user": "That works perfectly now, thank you! I have one last piece of code that's supposed to find the sum of all numbers in a list:\n\npython\n def find_sum(lst):\n sum = 0\n for num in lst:\n sum += num\n return num\n\nBut it's not returning the correct sum. Can you spot the issue?",
    "assistant": "Yes, I see the issue. In your return statement, you're returning num, which is the last number you're iterating over in the list, instead of sum, which holds the total sum. Here's the corrected code:\n\npython\n def find_sum(lst):\n sum = 0\n for num in lst:\n sum += num\n return sum\n"
    },
    ]
    chat_history = [elizabeth_chat_history, joe_chat_history, pierre_chat_history] 

    tasks = []
    async with aiohttp.ClientSession() as session:
        user = 0
        for user_history in chat_history:
            for _ in range(len(user_history)):
            # for _ in range(10):
                message = user_history[_]['user']
                llm_response = user_history[_]['assistant']
                user_id = users[user] 
                query = {
                    "message": message,
                    "llm_response": llm_response,
                    "user_id": user_id 
                    }
                tasks.append(fetch(session, url, query))        
            user += 1
        responses = await asyncio.gather(*tasks)
        
        for response, resp_time in responses:
            print(f"Response: {response}, Response Time: {resp_time} seconds")

# Run 
loop = asyncio.get_event_loop()
loop.run_until_complete(main())

