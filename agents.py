from typing import List, Union
import re

class ResearchAgent:
    def __init__(self, llm):
        self.llm = llm
        
    def research_topic(self, topic: str) -> str:
        prompt = f"""
        You are a research agent. Your job is to gather comprehensive information about: {topic}
        
        Provide detailed factual information, key concepts, and important aspects of this topic.
        Focus on accuracy and comprehensiveness.
        
        Research findings:
        """
        
        response = self.llm.invoke(prompt)
        return response

class AnalysisAgent:
    def __init__(self, llm):
        self.llm = llm
        
    def analyze_research(self, research_data: str, topic: str) -> str:
        prompt = f"""
        You are an analysis agent. Analyze the following research data about "{topic}":
        
        Research Data:
        {research_data}
        
        Your task:
        1. Identify key themes and patterns
        2. Extract the most important insights
        3. Highlight any contradictions or gaps
        4. Provide critical analysis
        
        Analysis:
        """
        
        response = self.llm.invoke(prompt)
        return response

class SummaryAgent:
    def __init__(self, llm):
        self.llm = llm
        
    def create_summary(self, research_data: str, analysis: str, topic: str) -> str:
        prompt = f"""
        You are a summary agent. Create a comprehensive summary report about "{topic}".
        
        Research Data:
        {research_data}
        
        Analysis:
        {analysis}
        
        Create a well-structured summary that includes:
        1. Executive Summary
        2. Key Findings
        3. Main Insights
        4. Conclusions
        5. Recommendations (if applicable)
        
        Final Report:
        """
        
        response = self.llm.invoke(prompt)
        return response