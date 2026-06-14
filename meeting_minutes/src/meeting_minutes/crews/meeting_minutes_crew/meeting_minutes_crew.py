from crewai import Agent, Crew, Process, Task, LLM
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import FileWriterTool

file_writer_tool_summary = FileWriterTool(file_name='summary.txt', directory='meeting_minutes')
file_writer_tool_action_items = FileWriterTool(file_name='action_items.txt', directory='meeting_minutes')
file_writer_tool_sentiment = FileWriterTool(file_name='sentiment.txt', directory='meeting_minutes')


@CrewBase
class MeetingMinutesCrew:
    """Meeting Minutes Crew"""

    agents: list[BaseAgent]
    tasks: list[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    ollama_llm = LLM(
        model="ollama/llama3.2:3b",
        base_url="http://localhost:11434",
    )

    @agent
    def meeting_minutes_summarizer(self) -> Agent:
        return Agent(
            config=self.agents_config["meeting_minutes_summarizer"],  # type: ignore[index]
            tools = [file_writer_tool_summary,file_writer_tool_action_items,file_writer_tool_sentiment],
            llm=self.ollama_llm
        )

    @agent
    def meeting_minutes_writer(self) -> Agent:
        return Agent(
            config=self.agents_config["meeting_minutes_writer"],  # type: ignore[index]
            llm=self.ollama_llm
        )


    @task
    def meeting_minutes_summary_task(self) -> Task:
        return Task(
            config=self.tasks_config["meeting_minutes_summary_task"],  # type: ignore[index]
        )

    @task
    def meeting_minutes_writing_task(self) -> Task:
        return Task(
            config=self.tasks_config["meeting_minutes_writing_task"],  # type: ignore[index]
        )



    @crew
    def crew(self) -> Crew:
        """Creates the meeting minutes Crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
