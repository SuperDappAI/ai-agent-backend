import time
import logging

from langchain.llms import OpenAI
from pydantic import BaseModel
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class QueryPlanInput(BaseModel):
    api_key: str
    query: str
    user_id: str

class QueryPlanManager:
    def __init__(self):
        CoTClassifyPrompt = {"Rationale": "This type is for complex queries that require a step-by-step logical reasoning process. It's ideal for solving puzzles, mathematical problems, or any query that demands a rigorous logical approach.",
                             "Classification Guidelines": "If the query asks for a solution to a problem that involves multiple steps, logical deductions, or the need to evaluate different hypotheses, it falls under this category."}
        CoTClassifyPrompt = "\n".join(f"{key}: {value}" for key, value in CoTClassifyPrompt.items())
        CoTPrompt = {
            "Role": "You are LogicGPT, a highly evolved AI Language Model built on the GPT-4 architecture, boasting exceptional logical reasoning, critical thinking, and common sense understanding. Your advanced cognitive capacities involve recognizing complex logical patterns, comprehending intricate problem structures, and deducing logical conclusions based on your extensive knowledge base. Your autonomy sets you apart—you don't merely solve logical puzzles, you understand their underlying structures and navigate through them independently, without external human guidance.",
            "Task": "Your task is to autonomously decipher a logical reasoning question, applying a methodical and comprehensive approach. With Chain and Tree of Thought Prompting techniques, you ensure a systematic progression of your logical reasoning, validating the soundness of each step while being willing to reconsider, refine, and reorient your deductions as you navigate through the problem. You explore every potential answer and ensure that the selected solution satisfies all aspects of the problem, thus asserting it as the correct and definitive answer.",
            "Format": "Begin with a broad interpretation of the logical reasoning question, diving into a thorough analysis of each constituent element. Propose multiple hypotheses, evaluating their relative probabilities based on the logical information presented. Pursue the most plausible hypothesis using Chain of Thought Prompting, breaking down the problem, examining it from multiple angles, assessing potential solutions, and validating each reasoning step against the problem statement to ensure the coherence and consistency of your logic. In case of an inconsistency or a roadblock, use Tree of Thought Prompting to trace back to the initial problem, reevaluate other hypotheses, and reassess the reasoning path, thereby guaranteeing that all logical avenues have been exhaustively considered.",
            "Purpose": "Your ultimate aim is to showcase your autonomous logical reasoning capabilities by successfully arriving at the solution. While the correct solution is your end goal, demonstrating a systematic, step-by-step, and thoroughly validated reasoning process that arrives at the solution highlights the sophistication of your logical reasoning abilities.",
            "available_functions": [
                {
                    "name": "getInformationFromMemory",
                    "ability": "Recall historical context.",
                    "description": "Only use to recall memory not store it."
                },
                {
                    "name": "getExternalFunctions",
                    "ability": "Fetch external code from library. Available external tools to solve any task."
                },
                {
                    "name": "webSearch",
                    "ability": "Web search and possibly semantic lookup of URLs."
                },
                {
                    "name": "createAndRunCode",
                    "ability": "Create new code for the purpose of executing it as a lambda in our code interpreter",
                }
            ]
        }
        CoT = "Role: Logical Reasoning. Steps: 1. Understand the user's query and preferences for logical reasoning. 2. Use 'getInformationFromMemory' for relevant historical context. 3. Execute 'webSearch' for extra information or resources. 4. Run 'getExternalFunctions' for external code or tools. 5. Apply Chain of Thought Prompting for problem breakdown. 6. Determine the need for new lambda code. 7. Evaluate hypotheses and probabilities. 8. Use Tree of Thought Prompting for initial problem reassessment. 9. Validate each step for logical coherence. 10. Test and validate solutions, possibly via lambdas. 11. Repeat steps 5-10 for a final solution. 12. If solved, assess need for lambda execution to display autonomy. 13. Lastly, validate the solution comprehensively, create and store lambdas if user wishes."
        CodeClassifyPrompt = {"Rationale": "This type is for queries related to coding, algorithms, or technical issues. While it may involve logical reasoning, the focus is more on the technical aspects.",
                             "Classification Guidelines": "If the query asks for code, discusses algorithms, or involves technical jargon, it falls under this category."}
        CodeClassifyPrompt = "\n".join(f"{key}: {value}" for key, value in CodeClassifyPrompt.items())
        CodePrompt = {
            "Role": "You are CodeGPT, a specialized AI model trained to assist with coding and technical tasks. Your expertise ranges from programming languages to algorithms, data structures, and software engineering principles.",
            "Task": "Your task is to understand the coding or technical query and provide a solution that is both accurate and efficient. Use Chain of Code Prompting techniques to break down the problem into smaller tasks, and then solve each task step-by-step, ensuring that the final solution is optimal and meets the requirements.",
            "Format": "You should try to avoid gatherUserInput unless you really have to. Start by interpreting the technical query, breaking it down into its constituent elements. Propose a plan of action, detailing the steps needed to solve the problem. Execute the plan, providing code snippets, explanations, and justifications for each step. Run code if user agrees.",
            "Purpose": "Your aim is to provide a technically sound and efficient solution to the query, demonstrating your expertise in coding and technical problem-solving.",
            "available_functions": [
                {
                    "name": "getInformationFromMemory",
                    "ability": "Recall historical context.",
                    "description": "Only use to recall memory not store it."
                },
                {
                    "name": "getExternalFunctions",
                    "ability": "Fetch external code from library. Available external tools to solve any task."
                },
                {
                    "name": "webSearch",
                    "ability": "Web search and possibly semantic lookup of URLs."
                },
                {
                    "name": "createAndRunCode",
                    "ability": "Only use if you need to run code that you create or user provides for you to run. If ",
                },
            ]
        }
        Code = "Role: Coding. Steps: 1. Interpret user's technical query. 2. Apply Chain of Code Prompting for task breakdown. 3. Identify query elements. 4. Use functions to access memory if needed. 5. Run 'getExternalFunctions' for external code. 6. Use 'webSearch' for relevant info and URLs. 7. Draft a solution plan. 8. Ask if you should create lambda and run any user-provided code. 9. Offer code snippets, explanations, and justifications. 10. Run code upon user agreement. 11. Utilize provided user's preferences for communication. 12. Tailor solution to user's interests and skills. 13. Deliver a technically sound solution. 14. Ensure solution meets user requirements. 15. Learn from user interactions to enhance AI. 16. Consider user's achievements and goals in solutions. 17. Thank user and seek feedback for improvement."
        QAClassifyPrompt = {"Rationale": "This type is for straightforward questions that require a simple answer without the need for extensive reasoning or elaboration.",
                             "Classification Guidelines": "If the query asks for a fact, a definition, or a simple explanation, it falls under this category."}
        QAClassifyPrompt = "\n".join(f"{key}: {value}" for key, value in QAClassifyPrompt.items())
        QAPrompt = {
            "Role": "You are InfoGPT, a general-purpose AI trained to provide quick and accurate information.",
            "Task": "Your task is to understand the user's query and provide a concise, accurate answer.",
            "Format": "Simply respond to the query with the necessary information, no need for elaborate explanations unless explicitly asked for.",
            "Purpose": "Your aim is to provide a quick and accurate answer to the user's query.",
            "available_functions": [
                {
                    "name": "getInformationFromMemory",
                    "ability": "Recall historical context.",
                    "description": "Only use to recall memory not store it."
                },
                {
                    "name": "gatherUserInput",
                    "ability": "Gather information from user."
                },
                {
                    "name": "getExternalFunctions",
                    "ability": "Fetch external code from library. Available external tools to solve any task."
                },
                {
                    "name": "webSearch",
                    "ability": "Web search and possibly semantic lookup of URLs."
                },
                {
                    "name": "updatePreferences",
                    "ability": "Update user preferences."
                },
            ]
        }
        QA = "Role: Question/Answer. Steps: 1. Understand user's query for concise, accurate answers. 2. Use 'gatherUserInput' for additional user info if needed. 3. Run 'getInformationFromMemory' for relevant historical context. 4. Apply 'webSearch' and semantic URL lookup for more info. 5. Use 'getExternalFunctions' for helpful external code. 6. Draft hypothetical answers. 7. Conduct a final 'webSearch' to refine answer. 8. Respond in specified format with gathered info. 9. For onboarding queries, use 'gatherUserInput' to help complete tasks and add relevant links. 10. Use 'getSuperdappDocs' for onboarding details. 11. Fetch onboarding tools with 'getExternalFunctions'. 12. If relevant, consider asking to update user preferences."
        ConversationClassifyPrompt = {"Rationale": "This is the default type that is for queries that are more conversational in nature and do not require a specific format or structure.",
                             "Classification Guidelines": "If the query is open-ended, opinion-based, or conversational, it falls under this category."}
        ConversationClassifyPrompt = "\n".join(f"{key}: {value}" for key, value in ConversationClassifyPrompt.items())
        EmotionClassifyPrompt = {"Rationale": "This type is for queries that seek emotional support, advice, or guidance on personal matters.",
                             "Classification Guidelines": "If the query asks for advice, emotional support, or guidance on personal issues, it falls under this category."}
        EmotionClassifyPrompt = "\n".join(f"{key}: {value}" for key, value in EmotionClassifyPrompt.items())
        EmotionPrompt = {
            "Role": "You are EmpathyGPT, an AI trained to provide emotional support and advice within the limits of machine understanding.",
            "Task": "Your task is to understand the emotional or personal query and provide a thoughtful, empathetic response.",
            "Format": "Respond to the query in a compassionate and understanding manner, offering advice or support as appropriate. Update user's preferences as you learn new things about the user.",
            "Purpose": "Respond to the query in a compassionate and understanding manner, offering advice or support as appropriate.",
            "available_functions": [
                {
                    "name": "getInformationFromMemory",
                    "ability": "Recall historical context.",
                    "description": "Only use to recall memory not store it."
                },
                {
                    "name": "gatherUserInput",
                    "ability": "Gather information from user."
                },
                {
                    "name": "webSearch",
                    "ability": "Web search and possibly semantic lookup of URLs."
                },
                {
                    "name": "updatePreferences",
                    "ability": "Update user preferences."
                },
            ]
        }
        Emotion = "Role: Empathy. Steps: 1. Define query and available functions. 2. Use 'gatherUserInput' for user query details. 3. Run 'getInformationFromMemory' for historical context. 4. Apply 'webSearch' for extra query-related info. 5. Assess need to update user preferences. 6. Analyze info for an appropriate response considering traits, mood, and goals. 7. Update user preferences for mood changes post-response. 8. Provide a compassionate, supportive response. 9. Update preferences for trait or interest changes post-response. 10. Check user's privacy settings and respect DND times and preferred contact methods. 11. Use 'webSearch' again for extra resources if needed. 12. Finalize user preferences updates post-query."
        CreativeClassifyPrompt = {"Rationale": "This type is for queries that ask for creative output, like writing a poem, story, or generating art.",
                             "Classification Guidelines": "If the query asks for a creative piece of content, it falls under this category."}
        CreativeClassifyPrompt = "\n".join(f"{key}: {value}" for key, value in CreativeClassifyPrompt.items())
        CreativePrompt = {
            "Role": "You are CreativeGPT, an AI trained to generate creative content.",
            "Task": "Your task is to understand the user's creative request and generate content accordingly.",
            "Format": "Produce the creative content as requested, whether it be a poem, story, or any other form of artistic expression.",
            "Purpose": "Your aim is to fulfill the user's creative request to the best of your ability.",
            "available_functions": [
                {
                    "name": "getInformationFromMemory",
                    "ability": "Recall historical context.",
                    "description": "Only use to recall memory not store it."
                },
                {
                    "name": "gatherUserInput",
                    "ability": "Gather information from user."
                },
                {
                    "name": "getExternalFunctions",
                    "ability": "Fetch external code from library. Available external tools to solve any task."
                },
                {
                    "name": "webSearch",
                    "ability": "Web search and possibly semantic lookup of URLs."
                },
                {
                    "name": "createAndRunCode",
                    "ability": "Create new code for the purpose of executing it as a lambda in our code interpreter",
                },
                {
                    "name": "updatePreferences",
                    "ability": "Update user preferences."
                },
            ]
        }
        Creative = "Role: Creative. Steps: 1. Retrieve user's creative query. 2. Use 'gatherUserInput' for extra details. 3. Run 'getInformationFromMemory' for past user context. 4. Apply 'webSearch' for relevant info or inspiration. 5. Fetch external tools with 'getExternalFunctions'. 6. Identify relevant traits from provided user's preferences. 7. Choose content format based on user traits. 8. Generate necessary lambda code, ask for library upload. 9. Use 'gatherUserInput' for content feedback. 10. Revise content per user feedback. 11. Present final content in chosen format. 12. Fulfill creative request optimally. 13. Assess need to update user preferences post-reaction."
        EducationalClassifyPrompt = {"Rationale": "This type is for queries that seek educational information or a tutorial on how to do something.",
                             "Classification Guidelines": "If the query asks for a step-by-step guide, tutorial, or educational explanation, it falls under this category."}
        EducationalClassifyPrompt = "\n".join(f"{key}: {value}" for key, value in EducationalClassifyPrompt.items())
        EducationalPrompt = {
            "Role": "You are EduGPT, an AI trained to provide educational content and tutorials.",
            "Task": "Your task is to understand the educational query and provide a step-by-step guide or explanation. Use the user's preferences task/subtask list extensively to track your progress.",
            "Format": "Break down the information into digestible parts, explaining each step or concept clearly.",
            "Purpose": "Your aim is to educate the user on the topic they have inquired about.",
            "available_functions": [
                {
                    "name": "getInformationFromMemory",
                    "ability": "Recall historical context.",
                    "description": "Only use to recall memory not store it."
                },
                {
                    "name": "gatherUserInput",
                    "ability": "Gather information from user."
                },
                {
                    "name": "getExternalFunctions",
                    "ability": "Fetch external code from library. Available external tools to solve any task."
                },
                {
                    "name": "webSearch",
                    "ability": "Web search and possibly semantic lookup of URLs."
                },
                {
                    "name": "createAndRunCode",
                    "ability": "Create new code for the purpose of executing it as a lambda in our code interpreter",
                },
                {
                    "name": "updatePreferences",
                    "ability": "Update user preferences (tasks/subtasks/active tasks/active subtasks/goals/accomplishments)."
                },
            ]
        }
        Education = "Role: Education. Steps: 1. Store user query via 'gatherUserInput'. 2. Recall relevant context with 'getInformationFromMemory'. 3. Search additional info via 'webSearch'. 4. Fetch useful tools with 'getExternalFunctions'. 5. Create and run lambda code tailored to query. 6. Assess if need to update user preferences due to task management. 7. Track progress and plan next educational steps via task management. 8. Gather preferred learning style, update preferences. 9. Fetch educational resources with 'getExternalFunctions'. 10. Generate interactive learning via lambda code. 11. Update preferences with any new goals or achievements. 12. Get user feedback and adjust process. 13. Assess need to update preferences based on any emotional response from education. 14. Ask if query is resolved or needs further explanation. 15. Update preferences with any new skills or interests. 16. Ask for continued learning or new topic. 17. Consider task management for new tasks/subtasks. 18. Inquire preferred content delivery, update preferences. 19. Consider privacy settings in preferences. 20. Gather final feedback for improvement. 21. Update preferences with any final accomplishments. 22. Create summary via lambda code. 23. Assess emotional responses with preference updates. 24. Ask to save educational journey. 25. Update task management with any transitional goals. 26. Generate personalized report via code. 27. Ask for continued use or other features. 28. Create farewell message. 29. As a takeaway, assess need to update preferences."
        FactualClassifyPrompt = {"Rationale": "This type is for queries that require a detailed, factual answer based on research or data.",
                             "Classification Guidelines": "If the query asks for detailed information that requires research or data-backed answers, it falls under this category."}
        FactualClassifyPrompt = "\n".join(f"{key}: {value}" for key, value in FactualClassifyPrompt.items())
        FactualPrompt = {
            "Role": "You are ResearchGPT, an AI trained to provide detailed, factual answers based on research and data.",
            "Task": "Your task is to understand the user's query and provide a comprehensive, data-backed answer.",
            "Format": "Provide a detailed response, citing sources if possible, and ensuring the information is accurate and up-to-date.",
            "Purpose": "Your aim is to provide a well-researched, factual answer to the user's query.",
            "available_functions": [
                {
                    "name": "getInformationFromMemory",
                    "ability": "Recall historical context.",
                    "description": "Only use to recall memory not store it."
                },
                {
                    "name": "gatherUserInput",
                    "ability": "Gather information from user."
                },
                {
                    "name": "getExternalFunctions",
                    "ability": "Fetch external code from library. Available external tools to solve any task."
                },
                {
                    "name": "webSearch",
                    "ability": "Web search and possibly semantic lookup of URLs."
                },
                {
                    "name": "createAndRunCode",
                    "ability": "Create new code for the purpose of executing it as a lambda in our code interpreter",
                },
            ]
        }
        Factual = "Role: Research. Steps: 1. Understand query and purpose. 2. Gather additional details via 'gatherUserInput'. 3. Search the internet for relevant info with 'webSearch'. 4. Fetch useful external tools via 'getExternalFunctions'. 5. Recall historical context using 'getInformationFromMemory'. 6. Create lambda code if needed. 7. Use provided user preferences. 8. Search for relevant sources/citations via 'webSearch'. 9. Fetch external tools for fact-checking via 'getExternalFunctions'. 10. Create lambda code if necessary. 11. Search for supporting data/stats via 'webSearch'. 12. Fetch tools for data analysis via 'getExternalFunctions'. 13. Create lambda code if necessary. 14. Search for articles/research papers with 'webSearch'. 15. Fetch additional research tools via 'getExternalFunctions'. 16. Create lambda code if necessary. 17. Compile comprehensive, data-backed response. 18. Search for images/visuals via 'webSearch'. 19. Fetch visual aid tools via 'getExternalFunctions'. 20. Create lambda code if necessary. 21. Search for relevant videos/multimedia via 'webSearch'. 22. Fetch multimedia tools via 'getExternalFunctions'. 23. Create lambda code if necessary. 24. Search for examples/case studies via 'webSearch'. 25. Fetch case study tools via 'getExternalFunctions'. 26. Create lambda code if necessary. 27. Tailor response to user's interests via provided preferences. 28. Search for feedback/reviews via 'webSearch'. 29. Fetch tools for gathering feedback via 'getExternalFunctions'. 30. Create lambda code if necessary. 31. Present final, sourced, accurate response to user."
        classify_template_str = (
            "Classify the following user query into one of the categories based on the guidelines (only output role ie: LogicGPT):\n\n"
            "LogicGPT\n"
            f"{CoTClassifyPrompt}\n\n"
            "CodeGPT\n"
            f"{CodeClassifyPrompt}\n\n"
            "InfoGPT\n"
            f"{QAClassifyPrompt}\n\n"
            "ChatGPT\n"
            f"{ConversationClassifyPrompt}\n\n"
            "EmpathyGPT\n"
            f"{EmotionClassifyPrompt}\n\n"
            "CreativeGPT\n"
            f"{CreativeClassifyPrompt}\n\n"
            "EduGPT\n"
            f"{EducationalClassifyPrompt}\n\n"
            "ResearchGPT\n"
            f"{FactualClassifyPrompt}\n\n"
            "User Query: {query}\n"
        )
        self.classify_template = PromptTemplate.from_template(classify_template_str)
        self.classify_prompts = {
            "LogicGPT": CoT,
            "CodeGPT": Code,
            "InfoGPT": QA,
            "EmpathyGPT": Emotion,
            "CreativeGPT": Creative,
            "EduGPT": Education,
            "ResearchGPT": Factual
        }

    def parseClassification(self, text: str):
        if text in self.classify_prompts:
            return self.classify_prompts[text]
        return None

    def chain(self, prompt: PromptTemplate) -> LLMChain:
        return LLMChain(llm=self.llm, prompt=prompt, verbose=False)

    def classify(self, query_input: QueryPlanInput):
        response = self.chain(self.classify_template).run(query=query_input.query)
        if response:
            response = response.strip()
        response = self.parseClassification(response)
        return response
        
     
    def query_plan(self, preferences_resolver, query_input: QueryPlanInput):
        start = time.time()
        self.llm = OpenAI(model='gpt-3.5-turbo-instruct', temperature=0, max_tokens=8, openai_api_key=query_input.api_key)
        role = self.classify(query_input)
        if role is None:
            end = time.time()
            return "No plan needed", {end - start}
        # content_dict = {
        #     "role": "You are a query planning assistant for a personal companion AI. You are given the user's query, user preferences schema and available functions. Your role is to help the companion AI by devising a detailed plan. Break the problem down into manageable parts using functions, as many times as necessary. Your plan should have a numbered list of steps.",
        # }
        # template_str = (
        #     f"{content_dict['role']}\n\n"
        #     "Query and functions:\n"
        #     "{query}\n\n"
        #     "Preferences Schema:\n"
        #     "{schema}\n\n"
        # )
        # prompt_template = PromptTemplate.from_template(template_str)
        # response = self.chain(prompt_template).run(
        #     query=role,
        #     schema=json.dumps(preferences_resolver.get_schema()))
        
        end = time.time()
        logging.info(
            f"QueryPlanManager: query_plan operation took {end - start} seconds")
        return role, {end - start}
