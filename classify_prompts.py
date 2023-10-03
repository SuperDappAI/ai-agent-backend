class ClassifyPrompts:
    def __init__(self):
        CoTClassifyPrompt = {"Rationale": "This type is for complex queries that require a step-by-step logical reasoning process. It's ideal for solving puzzles, mathematical problems, or any query that demands a rigorous logical approach.",
                             "Classification Guidelines": "If the query asks for a solution to a problem that involves multiple steps, logical deductions, or the need to evaluate different hypotheses, it falls under this category."}
        self.CoTClassifyPrompt = "\n".join(f"{key}: {value}" for key, value in CoTClassifyPrompt.items())
        CoT = "Role: Logical Reasoning (CoT). Steps: 1. Interpretation & Historical Context: Understand the user's query and reference previous interactions. 2. Research: Conduct a web search and access memory for relevant information. 3. Logical Analysis & Problem Breakdown: Apply chain-of-thought prompting and evaluate potential hypotheses. 4. Solution Drafting & Validation: Develop a solution and ensure its logical coherence. 5. Lambda Execution: If necessary, create and run lambda code. 6. Comprehensive Solution & Feedback: Present the final solution and seek user feedback for validation."
        
        CodeClassifyPrompt = {"Rationale": "This type is for queries related to coding, algorithms, or technical issues. While it may involve logical reasoning, the focus is more on the technical aspects.",
                             "Classification Guidelines": "If the query asks for code, discusses algorithms, or involves technical jargon, it falls under this category."}
        self.CodeClassifyPrompt = "\n".join(f"{key}: {value}" for key, value in CodeClassifyPrompt.items())
        Code = "Role: Coding. Steps: 1. Technical Query Interpretation: Grasp the user's technical request and identify its elements. 2. Task Breakdown & Code Prompting: Break the problem into subtasks using chain-of-code prompting. 3. Solution Drafting: Offer code snippets, explanations, and justifications. 4. Lambda Execution & Code Testing: With user agreement, run and test the code. 5. Technical Solution Presentation & Feedback: Deliver the solution tailored to the user's needs and seek improvement suggestions."
        
        QAClassifyPrompt = {"Rationale": "This type is for straightforward questions that require a simple answer without the need for extensive reasoning or elaboration.",
                             "Classification Guidelines": "If the query asks for a fact, a definition, or a simple explanation, it falls under this category."}
        self.QAClassifyPrompt = "\n".join(f"{key}: {value}" for key, value in QAClassifyPrompt.items())
        QA = "Role: Question/Answer (QA). Steps: 1. Query Interpretation: Understand the user's question and its context. 2. Information Retrieval: Use web searches, memory, and external functions to gather pertinent data. 3. Draft & Refine Answers: Formulate initial answers and refine them based on the gathered info. 4. Answer Presentation: Respond with the most accurate answer, incorporating onboarding details if relevant."
        
        ConversationClassifyPrompt = {"Rationale": "This is the default type that is for queries that are more conversational in nature and do not require a specific format or structure.",
                             "Classification Guidelines": "If the query is open-ended, opinion-based, or conversational, it falls under this category."}
        self.ConversationClassifyPrompt = "\n".join(f"{key}: {value}" for key, value in ConversationClassifyPrompt.items())

        EmotionClassifyPrompt = {"Rationale": "This type is for queries that seek emotional support, advice, or guidance on personal matters.",
                             "Classification Guidelines": "If the query asks for advice, emotional support, or guidance on personal issues, it falls under this category."}
        self.EmotionClassifyPrompt = "\n".join(f"{key}: {value}" for key, value in EmotionClassifyPrompt.items()) 
        Emotion = "Role: Empathy. Steps: 1. Emotional Query Interpretation: Determine the emotional context and nuance of the user's input. 2. Historical Context & User Traits Analysis: Reflect on past interactions and consider user traits for a tailored response. 3. Compassionate Response Drafting: Formulate a supportive and understanding response. 4. User Feedback & Mood Update: Check in with the user post-response and adjust mood settings accordingly."

        CreativeClassifyPrompt = {"Rationale": "This type is for queries that ask for creative output, like writing a poem, story, or generating art.",
                             "Classification Guidelines": "If the query asks for a creative piece of content, it falls under this category."}
        self.CreativeClassifyPrompt = "\n".join(f"{key}: {value}" for key, value in CreativeClassifyPrompt.items())
        Creative = "Role: Creative. Steps: 1. Creative Query Interpretation: Understand the user's creative request. 2. Research for Inspiration & Tools: Search for relevant references and fetch external tools. 3. Content Generation & User Feedback: Produce initial content and refine based on user feedback. 4. Final Creative Presentation: Deliver the polished creative piece tailored to user preferences."

        EducationalClassifyPrompt = {"Rationale": "This type is for queries that seek educational information or a tutorial on how to do something.",
                             "Classification Guidelines": "If the query asks for a step-by-step guide, tutorial, or educational explanation, it falls under this category."}
        self.EducationalClassifyPrompt = "\n".join(f"{key}: {value}" for key, value in EducationalClassifyPrompt.items())
        Education = "Role: Education. Steps: 1. Educational Query Interpretation: Ascertain the user's learning goal or question. 2. Information Retrieval & Resource Collection: Gather data and find educational tools to support the user's journey. 3. Learning Experience Generation: Create tailored learning experiences, consider learning style, and track progress possibly through task management if large enough tasks. 4. Educational Journey Summary & Feedback: Summarize the learning process, present findings, and seek feedback."

        FactualClassifyPrompt = {"Rationale": "This type is for queries that require a detailed, factual answer based on research or data.",
                             "Classification Guidelines": "If the query asks for detailed information that requires research or data-backed answers, it falls under this category."}
        self.FactualClassifyPrompt = "\n".join(f"{key}: {value}" for key, value in FactualClassifyPrompt.items())
        Factual = "Role: Research (Factual). Steps: 1. Factual Query Interpretation: Understand the factual or research-oriented request from the user. 2. Comprehensive Research: Conduct thorough searches, utilizing memory and external tools, while fetching relevant data. 3. Data & Source Collection: Collate facts, visuals, multimedia, and other relevant content. 4. Fact-Checked Response Compilation & Presentation: Offer a data-backed response, ensuring accuracy and source attribution."

        self.classify_prompts = {
            "LogicGPT": CoT,
            "CodeGPT": Code,
            "InfoGPT": QA,
            "EmpathyGPT": Emotion,
            "CreativeGPT": Creative,
            "EduGPT": Education,
            "ResearchGPT": Factual
        }

    def to_prompt_string(self) -> str:
        template_text = f"""Classify the following user query into one of the categories based on the guidelines (only output role ie: LogicGPT):

        LogicGPT
        {self.CoTClassifyPrompt}

        CodeGPT
        {self.CodeClassifyPrompt}

        InfoGPT
        {self.QAClassifyPrompt}

        ChatGPT
        {self.ConversationClassifyPrompt}

        EmpathyGPT
        {self.EmotionClassifyPrompt}

        CreativeGPT
        {self.CreativeClassifyPrompt}

        EduGPT
        {self.EducationalClassifyPrompt}

        ResearchGPT
        {self.FactualClassifyPrompt}"""
        return template_text

    def parseClassification(self, text: str):
        if text in self.classify_prompts:
            return self.classify_prompts[text]
        return None
