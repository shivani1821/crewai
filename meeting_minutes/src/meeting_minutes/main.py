from pathlib import Path
from pydantic import BaseModel
from crewai.flow import Flow, start,listen
from pydub import AudioSegment
from pydub.utils import make_chunks
from faster_whisper import WhisperModel
import os
from crews.meeting_minutes_crew.meeting_minutes_crew import MeetingMinutesCrew
from crews.gmailcrew.gmailcrew import GmailCrew

whisper_model = WhisperModel(
    "medium",
    device="cpu",
    compute_type="int8"
)


class MeetingMinutesState(BaseModel):
    transcript: str = ""
    meeting_minutes : str = ""


class MeetingMinutesFlow(Flow[MeetingMinutesState]):

    @start()
    def transcribe_meeting(self):
        print("Generating Transcription")

        script_dir = Path(__file__).parent
        audio_path = script_dir / "EarningsCall.wav"

        audio = AudioSegment.from_file(audio_path)

        chunk_length_ms = 60000
        chunks = make_chunks(audio, chunk_length_ms)

        full_transcription = ""

        for i, chunk in enumerate(chunks):
            print(f"Transcribing chunk {i + 1}/{len(chunks)}")

            chunk_path = script_dir / f"chunk_{i}.wav"
            chunk.export(chunk_path, format="wav")

            segments, _ = whisper_model.transcribe(
                str(chunk_path),
                language="en"
            )

            full_transcription += " ".join(
                segment.text for segment in segments
            ) + " "

            os.remove(chunk_path)

        self.state.transcript = full_transcription.strip()

        print(self.state.transcript)

        with open(
            script_dir / "transcript.txt",
            "w",
            encoding="utf-8"
        ) as f:
            f.write(self.state.transcript)

        return self.state.transcript

    @listen(transcribe_meeting)
    def generate_meeting_minutes(self):
        print("Generating meeting minutes")

        crew = MeetingMinutesCrew()

        inputs = {
            "transcript":self.state.transcript
        }
        meeting_minutes = crew.crew().kickoff(inputs)
        self.state.meeting_minutes = str(meeting_minutes)

    @listen(generate_meeting_minutes)
    def create_draft_meeting_minutes(self):
        print("Creating draft meeting minutes ")

        crew = GmailCrew()

        inputs = {
            "body": self.state.meeting_minutes
        }

        draft_crew = crew.crew().kickoff(inputs)
        print(f"Draft Crew: {draft_crew}")

def kickoff():
    meeting_minutes_flow = MeetingMinutesFlow()
    meeting_minutes_flow.plot()
    meeting_minutes_flow.kickoff()


if __name__ == "__main__":
    kickoff()