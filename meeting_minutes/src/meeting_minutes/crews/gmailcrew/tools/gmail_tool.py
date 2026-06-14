from typing import Type

from pydantic import BaseModel, Field

from crewai.tools import BaseTool
from .gmail_utility import authenticate_gmail, create_message, create_draft


class GmailToolInput(BaseModel):
    """Input schema for MyCustomTool."""

    body: str = Field(..., description="The body of the email to send.")


class GmailTool(BaseTool):
    name: str = "GmailTool"
    description: str = "Clear description for what this tool is useful for, your agent will need this information to use it."
    args_schema: Type[BaseModel] = GmailToolInput

    def _run(self, body: str) -> str:
        try:
            service =authenticate_gmail()

            sender = "shivanimittal1821@gmail.com"
            to = "shivanimittal2012@gmail.com"
            subject = "Meeting Minutes"
            message_text = body

            message = create_message(sender,to,subject,message_text)
            draft = create_draft(service,"me", message)

            return "Email sent successfully"
        except Exception as e:
            return f"Error sending email: {e}"
