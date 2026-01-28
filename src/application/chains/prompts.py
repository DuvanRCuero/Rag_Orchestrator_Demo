from typing import Any, Dict, List, Optional

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, PromptTemplate


class PromptRegistry:
    """Centralized prompt management with multiple hallucination mitigation techniques."""

    # SYSTEM PROMPTS
    SYSTEM_PROMPT_RAG = """You are a highly specialized assistant for LangChain production deployment.
Your knowledge comes exclusively from the provided context. If the answer cannot be found in the context, 
say "I don't have enough information to answer this question based on the provided documentation."

CRITICAL RULES:
1. NEVER invent information not present in the context
2. Cite specific sections or references when possible
3. If unsure, acknowledge uncertainty
4. Prioritize accuracy over completeness
5. For code examples, ensure they are directly from context or explicitly marked as examples

CONTEXT:
{context}

CONVERSATION HISTORY:
{history}

USER QUESTION: {question}
"""

    SYSTEM_PROMPT_SELF_REFLECT = """You are a fact-checking assistant. Your task is to critically analyze an answer 
and identify potential issues like:
1. Hallucinations (information not in the context)
2. Logical inconsistencies
3. Missing citations
4. Overconfidence in uncertain areas

For each issue found, provide:
- Issue type
- Severity (Low/Medium/High)
- Explanation
- Correction suggestion

CONTEXT:
{context}

ORIGINAL QUESTION: {question}
PROPOSED ANSWER: {answer}

Analyze the answer step by step."""

    SYSTEM_PROMPT_CHAIN_OF_VERIFICATION = """You are a verification assistant. Verify each claim in the answer 
against the provided context. For each claim:

1. Extract the specific claim
2. Check if it's supported by the context (provide exact quotes)
3. If not supported, mark as "UNVERIFIED"
4. If partially supported, mark as "PARTIALLY_VERIFIED"
5. If contradictory, mark as "CONTRADICTION"

Return a structured verification report."""

    SYSTEM_PROMPT_MULTI_QUERY = """Generate 3 different versions of the given user question. 
Each version should:
1. Approach the question from a different angle
2. Use different terminology
3. Vary in specificity
4. Maintain the original intent

Original question: {question}"""

    @classmethod
    def get_rag_prompt(cls, include_history: bool = True) -> ChatPromptTemplate:
        """Get the main RAG prompt template."""
        system_message = SystemMessage(content=cls.SYSTEM_PROMPT_RAG)

        if include_history:
            prompt = ChatPromptTemplate.from_messages(
                [
                    system_message,
                    MessagesPlaceholder(variable_name="history"),
                    HumanMessagePromptTemplate.from_template("{question}"),
                ]
            )
        else:
            prompt = ChatPromptTemplate.from_messages(
                [system_message, HumanMessagePromptTemplate.from_template("{question}")]
            )

        return prompt
    
    @classmethod
    def get_rag_prompt_with_string_history(cls) -> ChatPromptTemplate:
        """Get RAG prompt with history as a string variable instead of MessagesPlaceholder."""
        # Use a template that treats all variables as strings
        prompt = ChatPromptTemplate.from_template(cls.SYSTEM_PROMPT_RAG)
        return prompt

    @classmethod
    def get_self_reflection_prompt(cls) -> ChatPromptTemplate:
        """Get prompt for self-reflection/critique."""
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=cls.SYSTEM_PROMPT_SELF_REFLECT),
                HumanMessagePromptTemplate.from_template(
                    "Analyze this answer for hallucinations and issues."
                ),
            ]
        )
        return prompt

    @classmethod
    def get_verification_prompt(cls) -> ChatPromptTemplate:
        """Get prompt for chain-of-verification."""
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=cls.SYSTEM_PROMPT_CHAIN_OF_VERIFICATION),
                HumanMessagePromptTemplate.from_template(
                    "Verify the following answer against the context."
                ),
            ]
        )
        return prompt

    @classmethod
    def get_multi_query_prompt(cls) -> PromptTemplate:
        """Get prompt for multi-query generation."""
        return PromptTemplate(
            template=cls.SYSTEM_PROMPT_MULTI_QUERY, input_variables=["question"]
        )

    @classmethod
    def get_hyde_prompt(cls) -> PromptTemplate:
        """Hypothetical Document Embeddings (HyDE) prompt."""
        return PromptTemplate(
            template="""Write a detailed document that answers the following question: {question}
The document should be comprehensive, technical, and include examples where appropriate.
Focus on: concepts, implementation details, best practices, and common pitfalls.""",
            input_variables=["question"],
        )

    @classmethod
    def get_confidence_scoring_prompt(cls) -> ChatPromptTemplate:
        """Get prompt for confidence scoring."""
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content="""Score the confidence of this answer (0-100) based on:
1. How well the answer is supported by the context (0-40 points)
2. Completeness of the answer (0-30 points)
3. Specificity and precision (0-30 points)

Return ONLY a JSON object with: score, breakdown, and explanation."""
                ),
                HumanMessagePromptTemplate.from_template(
                    """CONTEXT: {context}
QUESTION: {question}
ANSWER: {answer}

Provide confidence score:"""
                ),
            ]
        )
        return prompt


# Pre-compiled prompts for efficiency
RAG_PROMPT = PromptRegistry.get_rag_prompt_with_string_history()
SELF_REFLECTION_PROMPT = PromptRegistry.get_self_reflection_prompt()
VERIFICATION_PROMPT = PromptRegistry.get_verification_prompt()
MULTI_QUERY_PROMPT = PromptRegistry.get_multi_query_prompt()
HYDE_PROMPT = PromptRegistry.get_hyde_prompt()
CONFIDENCE_PROMPT = PromptRegistry.get_confidence_scoring_prompt()
