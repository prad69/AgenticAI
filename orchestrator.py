import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from agents import ResearchAgent, AnalysisAgent, SummaryAgent

load_dotenv()

class MultiAgentOrchestrator:
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.7,
            model="gpt-3.5-turbo"
        )
        self.research_agent = ResearchAgent(self.llm)
        self.analysis_agent = AnalysisAgent(self.llm)
        self.summary_agent = SummaryAgent(self.llm)
    
    def run_research_pipeline(self, topic: str) -> dict:
        print(f"🔍 Starting research pipeline for: {topic}")
        
        print("\n📊 Research Agent working...")
        research_data = self.research_agent.research_topic(topic)
        
        print("🧠 Analysis Agent working...")
        analysis = self.analysis_agent.analyze_research(research_data, topic)
        
        print("📝 Summary Agent working...")
        summary = self.summary_agent.create_summary(research_data, analysis, topic)
        
        results = {
            "topic": topic,
            "research_data": research_data,
            "analysis": analysis,
            "final_summary": summary
        }
        
        print("✅ Pipeline completed!")
        return results
    
    def save_results(self, results: dict, filename: str = None):
        if filename is None:
            filename = f"research_report_{results['topic'].replace(' ', '_')}.txt"
        
        with open(filename, 'w') as f:
            f.write(f"RESEARCH REPORT: {results['topic']}\n")
            f.write("=" * 50 + "\n\n")
            f.write("RESEARCH DATA:\n")
            f.write(results['research_data'])
            f.write("\n\n" + "=" * 50 + "\n\n")
            f.write("ANALYSIS:\n")
            f.write(results['analysis'])
            f.write("\n\n" + "=" * 50 + "\n\n")
            f.write("FINAL SUMMARY:\n")
            f.write(results['final_summary'])
        
        print(f"💾 Results saved to {filename}")

if __name__ == "__main__":
    orchestrator = MultiAgentOrchestrator()
    
    topic = input("Enter a research topic: ")
    results = orchestrator.run_research_pipeline(topic)
    
    print("\n" + "=" * 60)
    print("FINAL SUMMARY:")
    print("=" * 60)
    print(results['final_summary'])
    
    save_option = input("\nSave full report to file? (y/n): ")
    if save_option.lower() == 'y':
        orchestrator.save_results(results)