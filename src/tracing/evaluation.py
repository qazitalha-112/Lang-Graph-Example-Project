"""Evaluation functions for task completion and quality assessment."""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import re
import json

from ..models.data_models import Task, TaskResult, TaskStatus
from ..config import AgentConfig


@dataclass
class EvaluationResult:
    """Result of an evaluation."""

    score: float  # 0.0 to 1.0
    max_score: float
    criteria: str
    details: Dict[str, Any]
    timestamp: datetime
    evaluator: str

    @property
    def percentage(self) -> float:
        """Get score as percentage."""
        return (self.score / self.max_score * 100) if self.max_score > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "score": self.score,
            "max_score": self.max_score,
            "percentage": self.percentage,
            "criteria": self.criteria,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "evaluator": self.evaluator,
        }


@dataclass
class TaskEvaluation:
    """Comprehensive evaluation of a task execution."""

    task_id: str
    task_type: str
    completion_score: EvaluationResult
    quality_score: EvaluationResult
    efficiency_score: EvaluationResult
    overall_score: float
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "completion_score": self.completion_score.to_dict(),
            "quality_score": self.quality_score.to_dict(),
            "efficiency_score": self.efficiency_score.to_dict(),
            "overall_score": self.overall_score,
            "recommendations": self.recommendations,
        }


class EvaluationManager:
    """
    Manages evaluation functions for task completion and quality assessment.

    This class provides:
    - Task completion evaluation
    - Output quality assessment
    - Efficiency metrics evaluation
    - Custom evaluation criteria
    - Automated evaluation workflows
    """

    def __init__(self, config: AgentConfig):
        """
        Initialize the evaluation manager.

        Args:
            config: Agent configuration
        """
        self.config = config
        self.custom_evaluators: Dict[str, Callable] = {}
        self.evaluation_history: List[TaskEvaluation] = []

    def evaluate_task_completion(
        self, task: Task, task_result: TaskResult
    ) -> EvaluationResult:
        """
        Evaluate whether a task was completed successfully.

        Args:
            task: The original task
            task_result: The result of task execution

        Returns:
            EvaluationResult for task completion
        """
        score = 0.0
        max_score = 1.0
        details = {}

        # Basic completion check
        if task_result.status == TaskStatus.COMPLETED.value:
            score += 0.4
            details["status_check"] = "passed"
        else:
            details["status_check"] = "failed"
            details["error"] = task_result.error_message

        # Success criteria evaluation
        if task.success_criteria:
            criteria_score = self._evaluate_success_criteria(
                task.success_criteria, task_result.output
            )
            score += criteria_score * 0.4
            details["criteria_score"] = criteria_score
            details["success_criteria"] = task.success_criteria

        # Artifact creation check
        if task_result.artifacts:
            score += 0.2
            details["artifacts_created"] = len(task_result.artifacts)
        else:
            details["artifacts_created"] = 0

        return EvaluationResult(
            score=score,
            max_score=max_score,
            criteria="task_completion",
            details=details,
            timestamp=datetime.now(),
            evaluator="completion_evaluator",
        )

    def evaluate_output_quality(
        self, task: Task, task_result: TaskResult
    ) -> EvaluationResult:
        """
        Evaluate the quality of task output.

        Args:
            task: The original task
            task_result: The result of task execution

        Returns:
            EvaluationResult for output quality
        """
        score = 0.0
        max_score = 1.0
        details = {}

        output = task_result.output or ""

        # Length and completeness check
        if len(output) > 50:  # Minimum meaningful output
            score += 0.2
            details["length_check"] = "passed"
        else:
            details["length_check"] = "failed"
            details["output_length"] = len(output)

        # Content quality based on task type
        task_type_score = self._evaluate_task_type_quality(task.task_type, output)
        score += task_type_score * 0.5
        details["task_type_score"] = task_type_score

        # Error absence check
        if not task_result.error_message:
            score += 0.2
            details["error_check"] = "passed"
        else:
            details["error_check"] = "failed"
            details["error_message"] = task_result.error_message

        # Artifact quality check
        artifact_score = self._evaluate_artifact_quality(task_result.artifacts)
        score += artifact_score * 0.1
        details["artifact_score"] = artifact_score

        return EvaluationResult(
            score=score,
            max_score=max_score,
            criteria="output_quality",
            details=details,
            timestamp=datetime.now(),
            evaluator="quality_evaluator",
        )

    def evaluate_efficiency(
        self, task: Task, task_result: TaskResult
    ) -> EvaluationResult:
        """
        Evaluate the efficiency of task execution.

        Args:
            task: The original task
            task_result: The result of task execution

        Returns:
            EvaluationResult for execution efficiency
        """
        score = 0.0
        max_score = 1.0
        details = {}

        # Execution time evaluation
        execution_time = task_result.execution_time
        time_score = self._evaluate_execution_time(task.task_type, execution_time)
        score += time_score * 0.4
        details["time_score"] = time_score
        details["execution_time"] = execution_time

        # Tool usage efficiency
        tool_efficiency = self._evaluate_tool_usage(
            task.required_tools, task_result.tool_usage
        )
        score += tool_efficiency * 0.3
        details["tool_efficiency"] = tool_efficiency
        details["tools_used"] = list(task_result.tool_usage.keys())

        # Resource utilization
        resource_score = self._evaluate_resource_usage(task_result)
        score += resource_score * 0.3
        details["resource_score"] = resource_score

        return EvaluationResult(
            score=score,
            max_score=max_score,
            criteria="efficiency",
            details=details,
            timestamp=datetime.now(),
            evaluator="efficiency_evaluator",
        )

    def evaluate_task_comprehensive(
        self, task: Task, task_result: TaskResult
    ) -> TaskEvaluation:
        """
        Perform comprehensive evaluation of a task.

        Args:
            task: The original task
            task_result: The result of task execution

        Returns:
            TaskEvaluation with all evaluation aspects
        """
        # Perform individual evaluations
        completion_eval = self.evaluate_task_completion(task, task_result)
        quality_eval = self.evaluate_output_quality(task, task_result)
        efficiency_eval = self.evaluate_efficiency(task, task_result)

        # Calculate overall score (weighted average)
        overall_score = (
            completion_eval.percentage * 0.5  # 50% weight on completion
            + quality_eval.percentage * 0.3  # 30% weight on quality
            + efficiency_eval.percentage * 0.2  # 20% weight on efficiency
        ) / 100.0

        # Generate recommendations
        recommendations = self._generate_recommendations(
            task, task_result, completion_eval, quality_eval, efficiency_eval
        )

        task_evaluation = TaskEvaluation(
            task_id=task.id,
            task_type=task.task_type,
            completion_score=completion_eval,
            quality_score=quality_eval,
            efficiency_score=efficiency_eval,
            overall_score=overall_score,
            recommendations=recommendations,
        )

        # Store in history
        self.evaluation_history.append(task_evaluation)

        return task_evaluation

    def _evaluate_success_criteria(self, criteria: str, output: str) -> float:
        """Evaluate how well the output meets the success criteria."""
        if not criteria or not output:
            return 0.0

        score = 0.0

        # Simple keyword matching for now
        # In a real implementation, this could use LLM-based evaluation
        criteria_lower = criteria.lower()
        output_lower = output.lower()

        # Check for key terms
        key_terms = [
            "complete",
            "successful",
            "finished",
            "done",
            "report",
            "analysis",
            "test",
            "result",
        ]

        matching_terms = sum(1 for term in key_terms if term in output_lower)
        score += min(matching_terms / len(key_terms), 1.0) * 0.5

        # Check for specific criteria keywords
        if "report" in criteria_lower and "report" in output_lower:
            score += 0.3
        if "test" in criteria_lower and (
            "test" in output_lower or "bug" in output_lower
        ):
            score += 0.3
        if "analysis" in criteria_lower and "analysis" in output_lower:
            score += 0.3

        return min(score, 1.0)

    def _evaluate_task_type_quality(self, task_type: str, output: str) -> float:
        """Evaluate output quality based on task type."""
        if not output:
            return 0.0

        output_lower = output.lower()

        if task_type == "web_testing":
            # Look for testing-related content
            testing_indicators = [
                "test",
                "bug",
                "issue",
                "functionality",
                "ui",
                "login",
            ]
            matches = sum(
                1 for indicator in testing_indicators if indicator in output_lower
            )
            return min(matches / len(testing_indicators), 1.0)

        elif task_type == "research":
            # Look for research-related content
            research_indicators = [
                "research",
                "information",
                "source",
                "finding",
                "data",
            ]
            matches = sum(
                1 for indicator in research_indicators if indicator in output_lower
            )
            return min(matches / len(research_indicators), 1.0)

        elif task_type == "analysis":
            # Look for analysis-related content
            analysis_indicators = [
                "analysis",
                "pattern",
                "recommendation",
                "insight",
                "conclusion",
            ]
            matches = sum(
                1 for indicator in analysis_indicators if indicator in output_lower
            )
            return min(matches / len(analysis_indicators), 1.0)

        else:
            # General quality check
            return 0.5 if len(output) > 100 else 0.2

    def _evaluate_execution_time(self, task_type: str, execution_time: float) -> float:
        """Evaluate execution time efficiency based on task type."""
        # Define expected time ranges for different task types
        time_expectations = {
            "web_testing": (30, 120),  # 30s to 2min
            "research": (20, 90),  # 20s to 1.5min
            "analysis": (15, 60),  # 15s to 1min
            "general": (10, 45),  # 10s to 45s
        }

        min_time, max_time = time_expectations.get(task_type, (10, 60))

        if execution_time <= min_time:
            return 1.0  # Very fast
        elif execution_time <= max_time:
            # Linear decrease from 1.0 to 0.5
            return 1.0 - (execution_time - min_time) / (max_time - min_time) * 0.5
        else:
            # Penalty for being too slow
            return max(0.1, 0.5 - (execution_time - max_time) / max_time * 0.4)

    def _evaluate_tool_usage(
        self, required_tools: List[str], actual_usage: Dict[str, int]
    ) -> float:
        """Evaluate efficiency of tool usage."""
        if not required_tools:
            return 1.0 if actual_usage else 0.5

        score = 0.0

        # Check if required tools were used
        used_required = sum(1 for tool in required_tools if tool in actual_usage)
        if required_tools:
            score += (used_required / len(required_tools)) * 0.6

        # Penalize excessive tool usage
        total_usage = sum(actual_usage.values())
        if total_usage <= 5:
            score += 0.4
        elif total_usage <= 10:
            score += 0.2
        # No bonus for excessive usage

        return min(score, 1.0)

    def _evaluate_resource_usage(self, task_result: TaskResult) -> float:
        """Evaluate resource usage efficiency."""
        score = 1.0

        # Penalize if no artifacts were created when they should have been
        if not task_result.artifacts:
            score -= 0.3

        # Bonus for creating multiple useful artifacts
        if len(task_result.artifacts) > 1:
            score += 0.2

        return max(0.0, min(score, 1.0))

    def _evaluate_artifact_quality(self, artifacts: List[str]) -> float:
        """Evaluate the quality of created artifacts."""
        if not artifacts:
            return 0.0

        score = 0.0

        # Bonus for having artifacts
        score += 0.5

        # Bonus for meaningful artifact names
        meaningful_extensions = [".md", ".json", ".txt", ".py", ".html"]
        meaningful_count = sum(
            1
            for artifact in artifacts
            if any(artifact.endswith(ext) for ext in meaningful_extensions)
        )

        if meaningful_count > 0:
            score += 0.5 * (meaningful_count / len(artifacts))

        return min(score, 1.0)

    def _generate_recommendations(
        self,
        task: Task,
        task_result: TaskResult,
        completion_eval: EvaluationResult,
        quality_eval: EvaluationResult,
        efficiency_eval: EvaluationResult,
    ) -> List[str]:
        """Generate recommendations based on evaluation results."""
        recommendations = []

        # Completion recommendations
        if completion_eval.percentage < 70:
            recommendations.append(
                "Consider breaking down complex tasks into smaller, more manageable subtasks"
            )
            if task_result.error_message:
                recommendations.append(
                    "Review error handling and add retry mechanisms for failed operations"
                )

        # Quality recommendations
        if quality_eval.percentage < 60:
            recommendations.append(
                "Improve output quality by providing more detailed and structured responses"
            )
            if len(task_result.output or "") < 100:
                recommendations.append(
                    "Ensure outputs are comprehensive and provide sufficient detail"
                )

        # Efficiency recommendations
        if efficiency_eval.percentage < 50:
            if task_result.execution_time > 120:
                recommendations.append(
                    "Optimize execution time by improving tool selection and usage patterns"
                )

            tool_count = sum(task_result.tool_usage.values())
            if tool_count > 10:
                recommendations.append(
                    "Reduce tool usage by selecting more appropriate tools for the task"
                )

        # Artifact recommendations
        if not task_result.artifacts:
            recommendations.append(
                "Consider creating artifacts to persist important results and findings"
            )

        return recommendations

    def register_custom_evaluator(
        self, name: str, evaluator_func: Callable[[Task, TaskResult], EvaluationResult]
    ) -> None:
        """
        Register a custom evaluation function.

        Args:
            name: Name of the evaluator
            evaluator_func: Function that takes Task and TaskResult and returns EvaluationResult
        """
        self.custom_evaluators[name] = evaluator_func

    def run_custom_evaluation(
        self, evaluator_name: str, task: Task, task_result: TaskResult
    ) -> Optional[EvaluationResult]:
        """
        Run a custom evaluation.

        Args:
            evaluator_name: Name of the registered evaluator
            task: The task to evaluate
            task_result: The task result to evaluate

        Returns:
            EvaluationResult from the custom evaluator, or None if not found
        """
        if evaluator_name in self.custom_evaluators:
            return self.custom_evaluators[evaluator_name](task, task_result)
        return None

    def get_evaluation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all evaluations performed.

        Returns:
            Dictionary containing evaluation statistics
        """
        if not self.evaluation_history:
            return {"message": "No evaluations performed yet"}

        total_evaluations = len(self.evaluation_history)

        # Calculate averages
        avg_completion = (
            sum(e.completion_score.percentage for e in self.evaluation_history)
            / total_evaluations
        )
        avg_quality = (
            sum(e.quality_score.percentage for e in self.evaluation_history)
            / total_evaluations
        )
        avg_efficiency = (
            sum(e.efficiency_score.percentage for e in self.evaluation_history)
            / total_evaluations
        )
        avg_overall = (
            sum(e.overall_score for e in self.evaluation_history) / total_evaluations
        )

        # Task type breakdown
        task_types = {}
        for evaluation in self.evaluation_history:
            task_type = evaluation.task_type
            if task_type not in task_types:
                task_types[task_type] = {"count": 0, "avg_score": 0.0}
            task_types[task_type]["count"] += 1
            task_types[task_type]["avg_score"] += evaluation.overall_score

        # Calculate averages for task types
        for task_type in task_types:
            task_types[task_type]["avg_score"] /= task_types[task_type]["count"]

        return {
            "total_evaluations": total_evaluations,
            "average_scores": {
                "completion": avg_completion,
                "quality": avg_quality,
                "efficiency": avg_efficiency,
                "overall": avg_overall * 100,  # Convert to percentage
            },
            "task_type_breakdown": task_types,
            "timestamp": datetime.now().isoformat(),
        }

    def export_evaluations(self, format_type: str = "json") -> str:
        """
        Export evaluation history.

        Args:
            format_type: Export format ("json" or "csv")

        Returns:
            Formatted evaluation data
        """
        if format_type.lower() == "json":
            return json.dumps(
                [eval.to_dict() for eval in self.evaluation_history], indent=2
            )
        elif format_type.lower() == "csv":
            lines = [
                "task_id,task_type,completion_score,quality_score,efficiency_score,overall_score"
            ]
            for evaluation in self.evaluation_history:
                lines.append(
                    f"{evaluation.task_id},{evaluation.task_type},"
                    f"{evaluation.completion_score.percentage:.2f},"
                    f"{evaluation.quality_score.percentage:.2f},"
                    f"{evaluation.efficiency_score.percentage:.2f},"
                    f"{evaluation.overall_score * 100:.2f}"
                )
            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

    def clear_evaluation_history(self) -> None:
        """Clear all evaluation history."""
        self.evaluation_history.clear()
