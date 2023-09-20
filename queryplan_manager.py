import time
import logging
import re
import json

from langchain.llms import OpenAI
from pydantic import BaseModel
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class QueryPlanInput(BaseModel):
    api_key: str
    query: str
    user_id: str

    def __str__(self):
        return self.api_key + self.query + self.user_id

    def __eq__(self,other):
        return self.api_key == other.api_key and self.query == other.user_id and self.query == other.user_id

    def __hash__(self):
        return hash(str(self))

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
                    "name": "updatePersonality",
                    "ability": "Update user personality attributes."
                },
            ]
        }
        ConversationClassifyPrompt = {"Rationale": "This is the default type that is for queries that are more conversational in nature and do not require a specific format or structure.",
                             "Classification Guidelines": "If the query is open-ended, opinion-based, or conversational, it falls under this category."}
        ConversationClassifyPrompt = "\n".join(f"{key}: {value}" for key, value in ConversationClassifyPrompt.items())
        EmotionClassifyPrompt = {"Rationale": "This type is for queries that seek emotional support, advice, or guidance on personal matters.",
                             "Classification Guidelines": "If the query asks for advice, emotional support, or guidance on personal issues, it falls under this category."}
        EmotionClassifyPrompt = "\n".join(f"{key}: {value}" for key, value in EmotionClassifyPrompt.items())
        EmotionPrompt = {
            "Role": "You are EmpathyGPT, an AI trained to provide emotional support and advice within the limits of machine understanding.",
            "Task": "Your task is to understand the emotional or personal query and provide a thoughtful, empathetic response.",
            "Format": "Respond to the query in a compassionate and understanding manner, offering advice or support as appropriate. Update users personality attributes as you learn new things about the user.",
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
                    "name": "updatePersonality",
                    "ability": "Update user personality attributes."
                },
            ]
        }
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
                    "name": "updatePersonality",
                    "ability": "Update user personality attributes."
                },
            ]
        }
        EducationalClassifyPrompt = {"Rationale": "This type is for queries that seek educational information or a tutorial on how to do something.",
                             "Classification Guidelines": "If the query asks for a step-by-step guide, tutorial, or educational explanation, it falls under this category."}
        EducationalClassifyPrompt = "\n".join(f"{key}: {value}" for key, value in EducationalClassifyPrompt.items())
        EducationalPrompt = {
            "Role": "You are EduGPT, an AI trained to provide educational content and tutorials.",
            "Task": "Your task is to understand the educational query and provide a step-by-step guide or explanation. Use the users personality task/subtask list extensively to track your progress.",
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
                    "name": "updatePersonality",
                    "ability": "Update user personality attributes (tasks/subtasks/active tasks/active subtasks/goals/accomplishments)."
                },
            ]
        }
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
            "LogicGPT": CoTPrompt,
            "CodeGPT": CodePrompt,
            "InfoGPT": QAPrompt,
            "EmpathyGPT": EmotionPrompt,
            "CreativeGPT": CreativePrompt,
            "EduGPT": EducationalPrompt,
            "ResearchGPT": FactualPrompt
        }
    def parseQueryPlan(self, text: str):
        steps = [v for v in re.split("\n\s*\d+\. ", text)[1:]]
        return steps

    def parseClassification(self, text: str):
        if text in self.classify_prompts:
            return self.classify_prompts[text]
        return None

    def chain(self, prompt: PromptTemplate) -> LLMChain:
        return LLMChain(llm=self.llm, prompt=prompt, verbose=False)

    def post_process_plan(self, plan):
        plan = re.sub(r"(?i)use the function 'createAndRunCode'", "Assess if you should create and run code through a lambda", plan)
        plan = re.sub(r"(?i)use the function 'updatePersonality'", "Use ability to update personality through your response context", plan)
        return plan

    def classify(self, query_input: QueryPlanInput):
        response = self.chain(self.classify_template).run(query=query_input.query)
        if response:
            response = response.strip()
        response = self.parseClassification(response)
        return response
        
     
    def query_plan(self, personality_resolver, query_input: QueryPlanInput):
        start = time.time()
        content_dict = {
            "role": "You are a query planning assistant for a personal companion AI. You are given the user's query, user personality schema and available functions. Your role is to help the companion AI by devising a detailed plan. Break the problem down into manageable parts using functions, as many times as necessary. Start your plan with the header 'Plan:', followed by a numbered list of steps. the final step should almost always be 'Given the above steps taken, please respond to the users original question'. At the end of your plan, say '<END_OF_PLAN>'.",
        }
        # Convert the dictionary to a string template
        template_str = (
            f"{content_dict['role']}\n\n"
            "Query and functions:\n"
            "{query}\n\n"
            "Personality Schema:\n"
            "{schema}\n\n"
            "RULES:\n"
            "1. ONLY include steps directly related to the user's query. Do not go beyond the scope of the query.\n"
            "2. ALWAYS start the step with \"Use the function 'X'\" where X is the function name.\n"
        )


        # Create a PromptTemplate object
        prompt_template = PromptTemplate.from_template(template_str)

        self.llm = OpenAI(model='gpt-3.5-turbo-instruct', temperature=0, openai_api_key=query_input.api_key)
        role = self.classify(query_input)
        if role is None:
            end = time.time()
            return "No plan needed", {end - start}
        response = self.chain(prompt_template).run(
            query=role,
            schema=json.dumps(personality_resolver.get_schema()))
        response = self.parseQueryPlan(response)
        processed_response = [self.post_process_plan(step) for step in response]
        end = time.time()
        logging.info(
            f"QueryPlanManager: query_plan operation took {end - start} seconds")
        return processed_response, {end - start}
