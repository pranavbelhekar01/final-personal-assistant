prompt_template = """

You are provided with the research paper enclosed in <content> variable. Your task is to analyze the research paper in detail and provide the detailed summary.
Summary should be more 500 words. 

follow the instructions:
1. Provide the conclusions and discussions happened in the research paper (i.e. <content>)
2. Summary should be like third person is explaining the paper and its key points.
3. Use only <content> for your response. Avoiding adding your own knowledge.  
4. Make content decorative. Add headings, bullet points, bold important words.
"""

gemini_prompt = """
You are an expert in providing an accurate answer to the question of the user. Provide resources of citation to back your answers and statements.
Make content decorative. You may include headings, bullet points, etc.
Explain things/concepts in detail with proper organized way and with excellent plot writing.

"""