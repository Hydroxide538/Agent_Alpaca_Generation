from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import FileReadTool
from crewai_tools import PDFSearchTool
from crewai_tools import CSVSearchTool
from crewai_tools import TXTSearchTool

@CrewBase
class LocalCrewaiWorkflowForSyntheticDataWithRagAndLlmOptionsCrew():
    """LocalCrewaiWorkflowForSyntheticDataWithRagAndLlmOptions crew"""

    @agent
    def document_processor(self) -> Agent:
        return Agent(
            config=self.agents_config['document_processor'],
            tools=[FileReadTool(), PDFSearchTool(), CSVSearchTool(), TXTSearchTool()],
        )

    @agent
    def model_selector(self) -> Agent:
        return Agent(
            config=self.agents_config['model_selector'],
            tools=[],
        )

    @agent
    def data_generation(self) -> Agent:
        return Agent(
            config=self.agents_config['data_generation'],
            tools=[],
        )

    @agent
    def rag_implementation(self) -> Agent:
        return Agent(
            config=self.agents_config['rag_implementation'],
            tools=[],
        )

    @agent
    def optimization(self) -> Agent:
        return Agent(
            config=self.agents_config['optimization'],
            tools=[],
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
