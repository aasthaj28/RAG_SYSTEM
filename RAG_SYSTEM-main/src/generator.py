"""
Generator Module
Handles answer generation using LLMs (OpenAI GPT models).
"""

import os
from typing import List, Dict, Any, Optional
from loguru import logger

try:
    from openai import OpenAI
except ImportError:
    print("Warning: OpenAI not installed. Install with: pip install openai")


class AnswerGenerator:
    """
    Generates answers using LLMs based on retrieved context.
    """

    # Strong, concise system prompt
    DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant.
You answer questions clearly, concisely, and only based on the given context.
If the context does not contain the answer, say: "I couldn't find that information in the documents.""""

    # Improved QA prompt template
    DEFAULT_QA_TEMPLATE = """You are a helpful and concise assistant.

Use ONLY the information inside the context to answer the user's question.

Context:
{context}

Question:
{question}

Rules for your answer:
- Answer clearly and directly (1 to 4 sentences max)
- Summarize in your own words
- DO NOT copy long paragraphs from context
- DO NOT hallucinate
- If answer not found in context → say: "I couldn't find that information in the documents."

Answer:
"""

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
        system_prompt: Optional[str] = None,
        qa_template: Optional[str] = None
    ):
        """
        Initialize Answer Generator.
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize OpenAI client
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable.")

        self.client = OpenAI(api_key=self.api_key)

        # Use improved prompt templates
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.qa_template = qa_template or self.DEFAULT_QA_TEMPLATE

        logger.info(f"AnswerGenerator initialized with model: {model}")

    def generate_answer(
        self,
        question: str,
        context: str,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Generate an answer based on the question and retrieved context.
        """
        logger.info(f"Generating answer for question: {question[:50]}...")

        # Format final prompt sent to model
        user_prompt = self.qa_template.format(
            context=context,
            question=question
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            answer = response.choices[0].message.content.strip()

            result = {
                'answer': answer,
                'question': question,
                'model': self.model
            }

            if include_sources:
                result['sources'] = self._extract_sources(context)

            return result

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                'answer': f"Error generating answer: {str(e)}",
                'question': question,
                'error': str(e)
            }

    def _extract_sources(self, context: str) -> List[Dict[str, str]]:
        """
        Extract source metadata from formatted context.
        """
        sources = []
        lines = context.split('\n')

        for line in lines:
            if line.startswith('[Document') and 'Source:' in line:
                try:
                    source_part = line.split('Source:')[1].strip().rstrip(')')
                    document_marker = line.split(']')[0] + ']'
                    sources.append({'source': source_part, 'document_marker': document_marker})
                except:
                    pass

        return sources
