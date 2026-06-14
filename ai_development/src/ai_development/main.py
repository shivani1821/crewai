#!/usr/bin/env python
import sys
import warnings

from datetime import datetime

from crew import AiDevelopment

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def run():
    """
    Run the crew.
    """
    inputs = {
        'topic': 'Artificial Intelligence LLMs ',
        'current_year': str(datetime.now().year)
    }

    try:
        AiDevelopment().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


run()