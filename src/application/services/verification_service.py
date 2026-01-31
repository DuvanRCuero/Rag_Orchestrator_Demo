"""Verification Service - handles answer verification and hallucination detection."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.domain.interfaces import LLMService
from src.application.chains.prompts import PromptRegistry


@dataclass
class VerificationResult:
    is_verified: bool
    confidence_score: float
    issues: List[Dict[str, Any]]
    verified_claims: List[str]


class VerificationService:

    def __init__(self, llm_service: LLMService):
        self.llm = llm_service

    async def verify_answer(
        self,
        query: str,
        answer: str,
        context: str,
    ) -> VerificationResult:
        messages = [
            {"role": "system", "content": PromptRegistry.SYSTEM_PROMPT_CHAIN_OF_VERIFICATION},
            {"role": "user", "content": f"Question: {query}\nAnswer: {answer}\nContext: {context}"},
        ]

        try:
            verification_text = await self.llm.generate(messages)
            return self._parse_verification(verification_text)
        except Exception:
            return VerificationResult(
                is_verified=True,
                confidence_score=0.7,
                issues=[],
                verified_claims=[],
            )

    async def self_reflect(
        self,
        query: str,
        answer: str,
        context: str,
    ) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": PromptRegistry.SYSTEM_PROMPT_SELF_REFLECT},
            {"role": "user", "content": f"Context: {context}\nQuestion: {query}\nAnswer: {answer}"},
        ]

        try:
            reflection = await self.llm.generate(messages)
            return self._parse_reflection(reflection)
        except Exception:
            return {"critique": "", "suggestions": [], "has_issues": False}

    async def calculate_confidence(
        self,
        query: str,
        answer: str,
        context: str,
    ) -> float:
        messages = [
            {"role": "system", "content": "Rate the confidence of this answer from 0 to 100. Return only a JSON object with a 'score' field."},
            {"role": "user", "content": f"Context: {context}\nQuestion: {query}\nAnswer: {answer}"},
        ]

        try:
            result = await self.llm.generate_json(messages)
            if isinstance(result, dict) and "score" in result:
                return float(result["score"]) / 100.0
            return 0.7
        except Exception:
            return 0.7

    def _parse_verification(self, text: str) -> VerificationResult:
        issues = []
        verified_claims = []

        for line in text.split("\n"):
            lower = line.lower()
            if "unverified" in lower or "contradiction" in lower:
                issues.append({"type": "unverified", "claim": line.strip()})
            elif "verified" in lower or "supported" in lower:
                verified_claims.append(line.strip())

        is_verified = len(issues) == 0
        confidence = 1.0 - (len(issues) / max(len(issues) + len(verified_claims), 1))

        return VerificationResult(
            is_verified=is_verified,
            confidence_score=confidence,
            issues=issues,
            verified_claims=verified_claims,
        )

    def _parse_reflection(self, text: str) -> Dict[str, Any]:
        has_issues = "issue" in text.lower() or "problem" in text.lower()
        return {
            "critique": text,
            "suggestions": [],
            "has_issues": has_issues,
        }
