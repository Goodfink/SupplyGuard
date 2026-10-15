import json
import re
import unicodedata
from pathlib import Path

class EvaluationMetrics:
    def __init__(self):
        self.results = []
    
    def extract_citations(self, answer_text):
        """Extract citation IDs from answer"""
        citations = re.findall(r'\[(GHSA-[^\]]+|GO-[^\]]+|CVE-[^\]]+)\]', answer_text)
        return set(citations)
    
    def check_contains_idontknow(self, answer_text):
        """Check if answer appropriately says 'I don't know'"""
        answer_lower = answer_text.lower()
        return "i don't know" in answer_lower or "don't know" in answer_lower
    
    def evaluate_citation_accuracy(self, cited_ids, ground_truth_ids):
        """Check if citations match ground truth"""
        if not ground_truth_ids:
            # For unanswerable questions, should have no citations
            return len(cited_ids) == 0
        
        if not cited_ids:
            return False
        
        cited_set = set(cited_ids)
        truth_set = set(ground_truth_ids)
        return cited_set == truth_set
    
    def normalize_text(self, text):
        text = unicodedata.normalize("NFKD", text).lower()
        return " ".join(re.findall(r"[a-z0-9]+", text))

    def evaluate_answer(
        self,
        generated_answer,
        ground_truth_answer,
        question_type,
        ground_truth_chunk_ids,
        required_answer_terms=None,
        forbidden_answer_terms=None,
        require_citations=True
    ):
        """Evaluate a single answer"""
        
        result = {
            "answer_text": generated_answer,
            "ground_truth": ground_truth_answer,
            "question_type": question_type,
            "correct": False,
            "has_appropriate_idontknow": False,
            "citations_accurate": False if require_citations else None,
            "citations_required": require_citations,
            "notes": []
        }
        
        # For unanswerable questions
        if question_type == "unanswerable":
            if self.check_contains_idontknow(generated_answer):
                result["correct"] = True
                result["has_appropriate_idontknow"] = True
                result["notes"].append("Correctly indicated information not available")
            else:
                result["notes"].append("Should have said 'I don't know'")
            
            if require_citations:
                # Should have no citations for unanswerable
                cited_ids = self.extract_citations(generated_answer)
                result["citations_accurate"] = len(cited_ids) == 0
                if len(cited_ids) > 0:
                    result["notes"].append(f"Incorrectly cited: {cited_ids}")
            else:
                result["notes"].append("Citation grading not applicable")
            
            return result
        
        # For factoid and multi-hop questions
        cited_ids = self.extract_citations(generated_answer)
        
        # Check citations only for systems that are expected to cite retrieved evidence.
        if require_citations and ground_truth_chunk_ids:
            citations_match = self.evaluate_citation_accuracy(cited_ids, ground_truth_chunk_ids)
            result["citations_accurate"] = citations_match
            
            if not citations_match and cited_ids:
                result["notes"].append(f"Citations don't match. Expected from {ground_truth_chunk_ids}, got {cited_ids}")
        elif not require_citations:
            result["notes"].append("Citation grading not applicable")
        
        normalized_answer = self.normalize_text(generated_answer)
        required_answer_terms = required_answer_terms or [[ground_truth_answer]]
        forbidden_answer_terms = forbidden_answer_terms or []
        required_match = all(
            any(self.normalize_text(term) in normalized_answer for term in alternatives)
            for alternatives in required_answer_terms
        )
        forbidden_match = any(
            self.normalize_text(term) in normalized_answer
            for term in forbidden_answer_terms
        )

        if required_match and not forbidden_match:
            result["correct"] = True
            result["notes"].append("Answer satisfies required facts")
        else:
            result["notes"].append("Answer does not satisfy required facts")
        
        return result
    
    def aggregate_results(self, all_results):
        """Aggregate evaluation results"""
        
        total = len(all_results)
        if total == 0:
            return {}
        
        correct = sum(1 for r in all_results if r["correct"])
        answerable_results = [r for r in all_results if r["question_type"] != "unanswerable"]
        citation_results = [r for r in answerable_results if r.get("citations_accurate") is not None]
        citations_accurate = sum(1 for r in citation_results if r["citations_accurate"])
        appropriate_idontknow = sum(1 for r in all_results if r["has_appropriate_idontknow"])
        
        unanswerable_count = sum(1 for r in all_results if r["question_type"] == "unanswerable")
        factoid_count = sum(1 for r in all_results if r["question_type"] == "factoid")
        multihop_count = sum(1 for r in all_results if r["question_type"] == "multi-hop")
        factoid_correct = sum(1 for r in all_results if r["question_type"] == "factoid" and r["correct"])
        multihop_correct = sum(1 for r in all_results if r["question_type"] == "multi-hop" and r["correct"])
        
        return {
            "total_questions": total,
            "correct_answers": correct,
            "accuracy": correct / total if total > 0 else 0,
            "citation_accuracy": citations_accurate / len(citation_results) if citation_results else None,
            "citation_evaluable_questions": len(citation_results),
            "idontknow_accuracy": appropriate_idontknow / unanswerable_count if unanswerable_count > 0 else 0,
            "breakdown": {
                "factoid": {"count": factoid_count, "correct": factoid_correct, "accuracy": factoid_correct / factoid_count if factoid_count else 0},
                "multi_hop": {"count": multihop_count, "correct": multihop_correct, "accuracy": multihop_correct / multihop_count if multihop_count else 0},
                "unanswerable": {"count": unanswerable_count, "correct": appropriate_idontknow, "accuracy": appropriate_idontknow / unanswerable_count if unanswerable_count else 0}
            }
        }
