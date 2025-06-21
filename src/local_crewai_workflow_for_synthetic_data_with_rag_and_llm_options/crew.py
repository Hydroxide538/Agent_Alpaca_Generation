from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import FileReadTool
from crewai_tools import PDFSearchTool
from crewai_tools import CSVSearchTool
from crewai_tools import TXTSearchTool
from typing import Optional, Dict, Any
import os

@CrewBase
class LocalCrewaiWorkflowForSyntheticDataWithRagAndLlmOptionsCrew():
    """LocalCrewaiWorkflowForSyntheticDataWithRagAndLlmOptions crew"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.workflow_config = config or {}
        self._setup_llms()
    
    def _setup_llms(self):
        """Setup LLMs based on configuration"""
        self.data_generation_llm = None
        self.embedding_llm = None
        self.reranking_llm = None
        
        if self.workflow_config:
            # Setup data generation LLM
            if self.workflow_config.get('data_generation_model'):
                self.data_generation_llm = self._create_llm(
                    self.workflow_config['data_generation_model'],
                    self.workflow_config
                )
            
            # Setup embedding LLM
            if self.workflow_config.get('embedding_model'):
                self.embedding_llm = self._create_llm(
                    self.workflow_config['embedding_model'],
                    self.workflow_config
                )
            
            # Setup reranking LLM
            if self.workflow_config.get('reranking_model'):
                self.reranking_llm = self._create_llm(
                    self.workflow_config['reranking_model'],
                    self.workflow_config
                )
    
    def _create_llm(self, model_spec: str, config: Dict[str, Any]) -> LLM:
        """Create LLM instance based on model specification"""
        try:
            provider, model_name = model_spec.split(":", 1)
            
            if provider == "openai":
                return LLM(
                    model=f"openai/{model_name}",
                    api_key=config.get('openai_api_key')
                )
            elif provider == "ollama":
                return LLM(
                    model=f"ollama/{model_name}",
                    base_url=config.get('ollama_url', 'http://localhost:11434')
                )
            else:
                raise ValueError(f"Unknown provider: {provider}")
        except Exception as e:
            print(f"Error creating LLM for {model_spec}: {str(e)}")
            return None

    @agent
    def document_processor(self) -> Agent:
        return Agent(
            config=self.agents_config['document_processor'],
            tools=[FileReadTool(), PDFSearchTool(), CSVSearchTool(), TXTSearchTool()],
            llm=self.data_generation_llm,
        )

    @agent
    def model_selector(self) -> Agent:
        return Agent(
            config=self.agents_config['model_selector'],
            tools=[],
            llm=self.data_generation_llm,
        )

    @agent
    def data_generation(self) -> Agent:
        return Agent(
            config=self.agents_config['data_generation'],
            tools=[],
            llm=self.data_generation_llm,
        )

    @agent
    def rag_implementation(self) -> Agent:
        return Agent(
            config=self.agents_config['rag_implementation'],
            tools=[],
            llm=self.embedding_llm or self.data_generation_llm,
        )

    @agent
    def optimization(self) -> Agent:
        return Agent(
            config=self.agents_config['optimization'],
            tools=[],
            llm=self.data_generation_llm,
        )


    @task
    def upload_and_read_documents(self) -> Task:
        return Task(
            config=self.tasks_config['upload_and_read_documents'],
            tools=[FileReadTool(), PDFSearchTool(), CSVSearchTool(), TXTSearchTool()],
        )

    @task
    def select_models(self) -> Task:
        return Task(
            config=self.tasks_config['select_models'],
            tools=[],
        )

    @task
    def generate_synthetic_data_for_each_document(self) -> Task:
        return Task(
            config=self.tasks_config['generate_synthetic_data_for_each_document'],
            tools=[],
        )

    @task
    def implement_rag_for_each_document(self) -> Task:
        return Task(
            config=self.tasks_config['implement_rag_for_each_document'],
            tools=[],
        )

    @task
    def optimize_performance(self) -> Task:
        return Task(
            config=self.tasks_config['optimize_performance'],
            tools=[],
        )


    @crew
    def crew(self) -> Crew:
        """Creates the LocalCrewaiWorkflowForSyntheticDataWithRagAndLlmOptions crew"""
        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
