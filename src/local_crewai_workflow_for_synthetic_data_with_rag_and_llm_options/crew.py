from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import FileReadTool
from crewai_tools import PDFSearchTool
from crewai_tools import CSVSearchTool
from crewai_tools import TXTSearchTool
from typing import Optional, Dict, Any
import os
import sys
import os
import json
import yaml
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import FileReadTool, PDFSearchTool, CSVSearchTool, TXTSearchTool
from typing import Optional, Dict, Any, List

# Add backend directory to path for safe LLM wrapper and LLMManager
backend_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'backend')
if backend_path not in sys.path:
    sys.path.append(backend_path)

# Import LLMManager and SafeLLMFactory
from backend.llm_manager import LLMManager
from backend.safe_llm_wrapper import CrewAICompatibleLLM

@CrewBase
class LocalCrewaiWorkflowForSyntheticDataWithRagAndLlmOptionsCrew():
    """LocalCrewaiWorkflowForSyntheticDataWithRagAndLlmOptions crew"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.workflow_config = config or {}
        self.llm_manager = LLMManager() # Initialize LLMManager
        self._setup_llms()
    
    def _setup_llms(self):
        """Setup LLMs based on configuration and LLMManager"""
        # The Manager Agent's LLM is explicitly set by the user config
        manager_model_spec = self.workflow_config.get('manager_model', 'ollama:llama3.3:latest')
        self.manager_llm = self._create_llm_from_spec(manager_model_spec, self.workflow_config)
        
        # Worker LLMs will be dynamically selected by the Manager Agent,
        # but we need a default for agents that might not be managed initially
        self.default_worker_llm = self._create_llm_from_spec(
            self.workflow_config.get('data_generation_model', 'ollama:llama3.3:latest'),
            self.workflow_config
        )
        
        # Embedding and Reranking LLMs are still directly configured for RAG system
        embedding_model_spec = self.workflow_config.get('embedding_model')
        self.embedding_llm = self._create_llm_from_spec(embedding_model_spec, self.workflow_config) if embedding_model_spec else None
        
        reranking_model_spec = self.workflow_config.get('reranking_model')
        self.reranking_llm = self._create_llm_from_spec(reranking_model_spec, self.workflow_config) if reranking_model_spec else None

    def _create_llm_from_spec(self, model_spec: str, config: Dict[str, Any]):
        """Create CrewAI LLM instance from a model specification using CrewAICompatibleLLM."""
        try:
            provider, model_name = model_spec.split(":", 1)
            
            if provider == "ollama":
                ollama_url = config.get('ollama_url', 'http://host.docker.internal:11434')
                return CrewAICompatibleLLM(model_spec, config)
            elif provider == "openai":
                api_key = config.get('openai_api_key')
                if not api_key:
                    raise ValueError(f"OpenAI API key is required for model {model_spec}")
                return CrewAICompatibleLLM(model_spec, config) # Use wrapper for consistency
            elif provider == "claude": # Placeholder for Claude
                api_key = config.get('claude_api_key') # Assuming a claude_api_key in config
                if not api_key:
                    raise ValueError(f"Claude API key is required for model {model_spec}")
                return CrewAICompatibleLLM(model_spec, config) # Use wrapper for consistency
            else:
                raise ValueError(f"Unknown provider: {provider}")
        except Exception as e:
            print(f"Error creating LLM for {model_spec}: {str(e)}")
            return None

    @agent
    def manager_agent(self) -> Agent:
        """The Manager Agent orchestrates the workflow and selects LLMs."""
        agent_config = self.agents_config['manager_agent'].copy()
        return Agent(
            config=agent_config,
            llm=self.manager_llm, # Manager's LLM is explicitly set
            tools=[], # Manager will use internal logic to select LLMs, not external tools for this
            verbose=True,
            allow_delegation=True, # Manager can delegate tasks
        )

    @agent
    def document_processor(self) -> Agent:
        """Agent responsible for processing documents and extracting raw content."""
        basic_tools = [FileReadTool()]
        if self.embedding_llm is not None:
            try:
                basic_tools.extend([PDFSearchTool(), CSVSearchTool(), TXTSearchTool()])
            except Exception as e:
                print(f"Warning: Could not initialize search tools due to embedding dependency: {str(e)}")
                print("Continuing with basic file reading tools only.")
        
        agent_config = self.agents_config['document_processor'].copy()
        return Agent(
            config=agent_config,
            tools=basic_tools,
            llm=self.default_worker_llm, # This will be overridden by Manager's dynamic selection
            verbose=True,
        )

    @agent
    def fact_extractor(self) -> Agent:
        """Agent specialized in extracting structured facts from document content."""
        agent_config = self.agents_config['fact_extractor'].copy()
        return Agent(
            config=agent_config,
            llm=self.default_worker_llm, # This will be overridden by Manager's dynamic selection
            verbose=True,
        )

    @agent
    def concept_extractor(self) -> Agent:
        """Agent specialized in extracting structured concepts from document content."""
        agent_config = self.agents_config['concept_extractor'].copy()
        return Agent(
            config=agent_config,
            llm=self.default_worker_llm, # This will be overridden by Manager's dynamic selection
            verbose=True,
        )

    @agent
    def qa_generator(self) -> Agent:
        """Agent responsible for generating high-quality Q&A pairs in Alpaca format."""
        agent_config = self.agents_config['qa_generator'].copy()
        return Agent(
            config=agent_config,
            llm=self.default_worker_llm, # This will be overridden by Manager's dynamic selection
            verbose=True,
        )

    @agent
    def quality_evaluator(self) -> Agent:
        """Agent responsible for evaluating the quality of generated Q&A pairs."""
        agent_config = self.agents_config['quality_evaluator'].copy()
        return Agent(
            config=agent_config,
            llm=self.default_worker_llm, # This will be overridden by Manager's dynamic selection
            verbose=True,
        )

    @agent
    def llm_tester(self) -> Agent:
        """Agent responsible for running LLM "shoot-out" tests and recording performance."""
        agent_config = self.agents_config['llm_tester'].copy()
        return Agent(
            config=agent_config,
            llm=self.default_worker_llm, # This will be overridden by Manager's dynamic selection
            verbose=True,
        )

    @task
    def process_documents_task(self) -> Task:
        return Task(
            config=self.tasks_config['process_documents_task'],
            tools=[FileReadTool(), PDFSearchTool(), CSVSearchTool(), TXTSearchTool()], # Tools for document reading
            agent=self.document_processor(),
        )

    @task
    def extract_facts_task(self) -> Task:
        return Task(
            config=self.tasks_config['extract_facts_task'],
            agent=self.fact_extractor(),
        )

    @task
    def extract_concepts_task(self) -> Task:
        return Task(
            config=self.tasks_config['extract_concepts_task'],
            agent=self.concept_extractor(),
        )

    @task
    def generate_qa_pairs_task(self) -> Task:
        return Task(
            config=self.tasks_config['generate_qa_pairs_task'],
            agent=self.qa_generator(),
        )

    @task
    def evaluate_qa_quality_task(self) -> Task:
        return Task(
            config=self.tasks_config['evaluate_qa_quality_task'],
            agent=self.quality_evaluator(),
        )

    @task
    def run_llm_shootout_task(self) -> Task:
        return Task(
            config=self.tasks_config['run_llm_shootout_task'],
            agent=self.llm_tester(),
        )

    @crew
    def crew(self) -> Crew:
        """Creates the LocalCrewaiWorkflowForSyntheticDataWithRagAndLlmOptions crew"""
        return Crew(
            agents=[
                self.manager_agent(),
                self.document_processor(),
                self.fact_extractor(),
                self.concept_extractor(),
                self.qa_generator(),
                self.quality_evaluator(),
                self.llm_tester(),
            ],
            tasks=[
                self.process_documents_task(),
                self.extract_facts_task(),
                self.extract_concepts_task(),
                self.generate_qa_pairs_task(),
                self.evaluate_qa_quality_task(),
                self.run_llm_shootout_task(),
            ],
            process=Process.hierarchical, # Changed to hierarchical
            manager_llm=self.manager_llm, # Assign the manager LLM
            verbose=True,
        )
