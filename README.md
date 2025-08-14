# Multi-Agent AI Research System

A simple multi-agent AI system built with LangChain that demonstrates how different AI agents can work together to research, analyze, and summarize information.

## Project Structure

- `agents.py` - Contains three specialized agents:
  - **ResearchAgent**: Gathers comprehensive information about topics
  - **AnalysisAgent**: Analyzes research data and identifies patterns
  - **SummaryAgent**: Creates structured final reports
- `orchestrator.py` - Coordinates the agents and manages the research pipeline
- `requirements.txt` - Python dependencies
- `.env.example` - Environment variables template

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your OpenAI API key:
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

## Usage

Run the research pipeline:
```bash
python orchestrator.py
```

The system will:
1. Ask for a research topic
2. Use the Research Agent to gather information
3. Use the Analysis Agent to analyze the data
4. Use the Summary Agent to create a final report
5. Optionally save the full report to a file

## How It Works

This project demonstrates key agentic AI concepts:

- **Agent Specialization**: Each agent has a specific role and expertise
- **Sequential Processing**: Agents work in a pipeline, building on each other's outputs
- **Orchestration**: A central coordinator manages the workflow
- **State Management**: Information flows between agents in a structured way

## Example Topics to Try

- "Machine Learning in Healthcare"
- "Renewable Energy Technologies"
- "Blockchain Applications"
- "Artificial Intelligence Ethics"

## Learning Objectives

- Understanding multi-agent systems
- Agent coordination and communication
- LangChain framework basics
- Prompt engineering for different agent roles