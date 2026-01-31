"""Ask Question Use Case - orchestrates the full RAG pipeline."""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from src.application.services.retrieval_service import RetrievalService, RetrievalResult
from src.application.services.generation_service import GenerationService, GenerationResult
from src.application.services.verification_service import VerificationService
from src.application.services.query_enhancement_service import QueryEnhancementService
from src.application.chains.memory import EnhancedConversationMemory
from src.core.schemas import DocumentChunk


@dataclass
class AskQuestionInput:
    query: str
    session_id: Optional[str] = None
    top_k: int = 5
    use_multi_query: bool = True
    use_hyde: bool = False
    use_verification: bool = True


@dataclass
class AskQuestionOutput:
    answer: str
    sources: List[DocumentChunk]
    confidence: Optional[float]
    query_time: float
    session_id: str
    token_usage: Dict[str, int]
    verification_report: Optional[Dict[str, Any]] = None


class AskQuestionUseCase:

    def __init__(
        self,
        retrieval_service: RetrievalService,
        generation_service: GenerationService,
        verification_service: VerificationService,
        query_enhancement_service: QueryEnhancementService,
        conversation_memory: EnhancedConversationMemory,
    ):
        self.retrieval = retrieval_service
        self.generation = generation_service
        self.verification = verification_service
        self.query_enhancement = query_enhancement_service
        self.memory = conversation_memory

    async def execute(self, input: AskQuestionInput) -> AskQuestionOutput:
        session_id = input.session_id or self._generate_session_id()

        # Get conversation history
        history = ""
        if session_id:
            history = self.memory.get_conversation_for_context(session_id)

        # Query enhancement
        query_variations = []
        if input.use_multi_query:
            query_variations = await self.query_enhancement.generate_query_variations(input.query)

        # Retrieval
        if query_variations:
            retrieval_result = await self.retrieval.retrieve_with_multi_query(
                query=input.query,
                query_variations=query_variations,
                top_k=input.top_k,
            )
        else:
            retrieval_result = await self.retrieval.retrieve(
                query=input.query,
                top_k=input.top_k,
            )

        # Generation
        generation_result = await self.generation.generate(
            query=input.query,
            context_chunks=retrieval_result.chunks,
            history=history,
        )

        # Verification
        verification_report = None
        confidence = generation_result.confidence_score
        if input.use_verification:
            context = self._format_context(retrieval_result.chunks)
            verification = await self.verification.verify_answer(
                query=input.query,
                answer=generation_result.answer,
                context=context,
            )
            confidence = verification.confidence_score
            verification_report = {
                "is_verified": verification.is_verified,
                "issues": verification.issues,
            }

        # Save to memory
        self.memory.add_message(session_id, "user", input.query)
        self.memory.add_message(session_id, "assistant", generation_result.answer)

        return AskQuestionOutput(
            answer=generation_result.answer,
            sources=retrieval_result.chunks,
            confidence=confidence,
            query_time=retrieval_result.retrieval_time + generation_result.generation_time,
            session_id=session_id,
            token_usage=generation_result.token_usage,
            verification_report=verification_report,
        )

    def _generate_session_id(self) -> str:
        import uuid
        return str(uuid.uuid4())

    def _format_context(self, chunks: List[DocumentChunk]) -> str:
        return "\n\n".join([c.content for c in chunks])
