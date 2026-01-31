"""Stream Answer Use Case."""

from dataclasses import dataclass
from typing import Optional, List, AsyncIterator

from src.application.services.retrieval_service import RetrievalService
from src.application.services.generation_service import GenerationService
from src.application.chains.memory import EnhancedConversationMemory
from src.core.schemas import DocumentChunk


@dataclass
class StreamInput:
    query: str
    session_id: Optional[str] = None
    top_k: int = 5


class StreamAnswerUseCase:

    def __init__(
        self,
        retrieval_service: RetrievalService,
        generation_service: GenerationService,
        conversation_memory: EnhancedConversationMemory,
    ):
        self.retrieval = retrieval_service
        self.generation = generation_service
        self.memory = conversation_memory

    async def execute(self, input: StreamInput) -> tuple:
        # Retrieve first
        retrieval_result = await self.retrieval.retrieve(
            query=input.query,
            top_k=input.top_k,
        )

        # Get history
        history = ""
        if input.session_id:
            history = self.memory.get_conversation_for_context(input.session_id)

        # Return retrieval result and stream generator
        async def stream_generator() -> AsyncIterator[str]:
            full_answer = ""
            async for token in self.generation.stream(
                query=input.query,
                context_chunks=retrieval_result.chunks,
                history=history,
            ):
                full_answer += token
                yield token

            # Save to memory after streaming completes
            if input.session_id:
                self.memory.add_message(input.session_id, "user", input.query)
                self.memory.add_message(input.session_id, "assistant", full_answer)

        return retrieval_result, stream_generator()
