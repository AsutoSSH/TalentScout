import sys
from pathlib import Path

# Add the project root (2 levels up from this file) to Python's paths
sys.path.append(str(Path(__file__).resolve().parents[2]))


from pdfminer.high_level import extract_text
from pydantic import BaseModel, Field
from typing import List, Optional, Union
# from langchain_core.prompts import PromptTemplate
# from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os
from backend.core.agent_base import AgentBase

load_dotenv()


EXTRACTION_MODEL = os.getenv("EXTRACTION_MODEL")

# Helper fn to extract text from Resume pdfs
def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text from a PDF resume using pdfminer.six.
    More accurate than PyPDF2 for complex resume layouts.
    """
    try:
        text = extract_text(pdf_path)
        return text.strip()
    except Exception as e:
        raise RuntimeError(f"Error extracting text with pdfminer.six: {e}")
    

class Profile(BaseModel):
    extracted_skills: List[str] = Field(description="List of key technical skills")
    extracted_domains: List[str] = Field(description="List of industry domains")
    years_of_experience: Union[str, int] = Field(description="Experience in years or 'Unknown'")

class ProfileExtractorAgent(AgentBase):
    async def execute(self, text: str) -> Profile:
        # Construct the structured prompt using chat history format
        chat_history = [
            {
                "role": "system",
                "content": "You are an expert HR assistant tasked with extracting structured information from resume text."
            },
            {
                "role": "user",
                "content": f"Extract structured info from resume text:\n\n{text}"
            }
        ]
        
        # Call the parent model method passing the Profile response_model
        # This will route directly to Case 1 (instructor mode) in get_completion
        result = await self.call_model(
            chat_history=chat_history,
            response_model=Profile,
            temperature=0.0
        )
        
        # result.parsed contains the validated Pydantic object
        return result.parsed





import asyncio

# Create an async execution block at the bottom of your file
async def run_pipeline():
    resume_text = extract_text_from_pdf("Resume_1_1.pdf")
    
    # Initialize the agent properly with arguments your AgentBase expects
    agent = ProfileExtractorAgent(name="ResumeExtractor", provider=EXTRACTION_MODEL, verbose=False)
    
    result = await agent.execute(resume_text)
    print(result.model_dump())

if __name__ == "__main__":
    asyncio.run(run_pipeline())