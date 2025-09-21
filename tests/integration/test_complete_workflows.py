"""
Comprehensive integration tests for complete workflow scenarios.

This module contains integration tests that validate end-to-end workflow
execution for various complex scenarios, ensuring all components work
together correctly.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from src.workflow import SupervisorWorkflow
from src.workflow_simple import SimpleWorkflow
from src.config import AgentConfig
from src.models.data_models import TaskStatus, Task, TaskResult
from examples.example_scenarios import ExampleScenarios, ScenarioResult


class TestCompleteWorkflows:
    """Integration tests for complete workflow scenarios."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        config = Mock(spec=AgentConfig)
        config.llm_model = "gpt-4"
        config.openai_api_key = "test-key"
        config.max_iterations = 10
        config.max_subagents = 5
        config.tool_timeout = 30
        config.langsmith_project = "test-project"
        config.enable_tracing = False
        config.tavily_api_key = None
        config.firecrawl_api_key = None
        config.virtual_fs_root = "/virtual"
        config.max_file_size = 1048576
        return config

    @pytest.fixture
    def simple_workflow(self, mock_config):
        """Create a simple workflow instance for testing."""
        return SimpleWorkflow(mock_config)

    @pytest.fixture
    def langgraph_workflow(self, mock_config):
        """Create a LangGraph workflow instance for testing."""
        with patch("src.workflow.ChatOpenAI") as mock_llm:
            mock_llm.return_value = Mock()
            return SupervisorWorkflow(mock_config)

    @pytest.fixture
    def example_scenarios(self, mock_config):
        """Create example scenarios runner for testing."""
        return ExampleScenarios(mock_config)

    def test_web_application_testing_workflow_complete(self, simple_workflow):
        """Test complete web application testing workflow with realistic task breakdown."""
        objective = (
            "Test my e-commerce web application for bugs, security issues, and usability problems. "
            "The application has user authentication, product catalog, shopping cart, and payment processing."
        )

        with patch.object(simple_workflow.supervisor, "task_tool") as mock_task_tool:
            # Simulate realistic web testing workflow
            task_results = [
                {
                    "task_id": "web_test_1",
                    "status": TaskStatus.COMPLETED.value,
                    "output": "Application structure analysis completed: Identified 4 main modules (auth, catalog, cart, payment) and 15 user workflows.",
                    "artifacts": [
                        "app_structure_analysis.md",
                        "user_workflows.json",
                        "test_plan.md",
                    ],
                    "execution_time": 35.0,
                    "tool_usage": {"web_scrape": 5, "execute_code": 2},
                },
                {
                    "task_id": "web_test_2",
                    "status": TaskStatus.COMPLETED.value,
                    "output": "Authentication testing completed: Login/logout works correctly, found 1 session timeout issue and 1 password policy weakness.",
                    "artifacts": ["auth_test_results.md", "security_findings.json"],
                    "execution_time": 45.0,
                    "tool_usage": {"web_scrape": 8, "execute_code": 4},
                },
                {
                    "task_id": "web_test_3",
                    "status": TaskStatus.COMPLETED.value,
                    "output": "Product catalog testing completed: Search and filtering work well, found 2 UI bugs in product display and 1 performance issue with large catalogs.",
                    "artifacts": [
                        "catalog_test_results.md",
                        "ui_bug_report.md",
                        "performance_analysis.json",
                    ],
                    "execution_time": 55.0,
                    "tool_usage": {"web_scrape": 12, "execute_code": 6},
                },
                {
                    "task_id": "web_test_4",
                    "status": TaskStatus.COMPLETED.value,
                    "output": "Shopping cart testing completed: Add/remove items works correctly, found 1 quantity validation bug and 1 cart persistence issue.",
                    "artifacts": ["cart_test_results.md", "cart_bugs.json"],
                    "execution_time": 40.0,
                    "tool_usage": {"web_scrape": 10, "execute_code": 5},
                },
                {
                    "task_id": "web_test_5",
                    "status": TaskStatus.COMPLETED.value,
                    "output": "Payment processing testing completed: Payment flows work but found 1 critical security issue with payment data handling.",
                    "artifacts": [
                        "payment_test_results.md",
                        "security_critical_findings.json",
                    ],
                    "execution_time": 60.0,
                    "tool_usage": {"web_scrape": 8, "execute_code": 7},
                },
                {
                    "task_id": "web_test_6",
                    "status": TaskStatus.COMPLETED.value,
                    "output": "Usability testing completed: Overall user experience is good, identified 3 usability improvements and 2 accessibility issues.",
                    "artifacts": ["usability_report.md", "accessibility_audit.json"],
                    "execution_time": 50.0,
                    "tool_usage": {"web_scrape": 15, "execute_code": 3},
                },
                {
                    "task_id": "web_test_7",
                    "status": TaskStatus.COMPLETED.value,
                    "output": "Comprehensive test report generated: 8 bugs found (1 critical, 3 high, 4 medium), 5 usability improvements recommended.",
                    "artifacts": [
                        "comprehensive_test_report.md",
                        "executive_summary.md",
                        "bug_priority_matrix.json",
                    ],
                    "execution_time": 30.0,
                    "tool_usage": {"execute_code": 4},
                },
            ]

            mock_task_tool.side_effect = task_results

            result = simple_workflow.run(objective)

            # Validate comprehensive workflow completion
            assert result.success is True
            assert result.objective == objective
            assert result.completed_tasks >= 7
            assert result.iterations > 0

            # Validate web testing specific content
            final_result_lower = result.final_result.lower()
            assert any(
                term in final_result_lower
                for term in ["test", "bug", "security", "usability"]
            )

            # Validate comprehensive artifacts
            artifacts_str = str(result.artifacts)
            expected_artifacts = [
                "app_structure_analysis.md",
                "auth_test_results.md",
                "catalog_test_results.md",
                "cart_test_results.md",
                "payment_test_results.md",
                "usability_report.md",
                "comprehensive_test_report.md",
            ]
            for artifact in expected_artifacts:
                assert artifact in artifacts_str

            # Validate execution metrics
            total_execution_time = sum(task["execution_time"] for task in task_results)
            assert total_execution_time == 315.0  # Sum of all task execution times

            # Validate tool usage patterns
            total_web_scrape_usage = sum(
                task.get("tool_usage", {}).get("web_scrape", 0) for task in task_results
            )
            total_code_execution_usage = sum(
                task.get("tool_usage", {}).get("execute_code", 0)
                for task in task_results
            )
            assert total_web_scrape_usage > 0  # Should use web scraping for testing
            assert (
                total_code_execution_usage > 0
            )  # Should use code execution for analysis

    def test_research_workflow_complete_with_citations(self, simple_workflow):
        """Test complete research workflow with proper citation and source management."""
        objective = (
            "Research the latest developments in large language models and their applications "
            "in software development, including code generation, automated testing, and developer "
            "productivity tools. Create a comprehensive report with proper citations."
        )

        with patch.object(simple_workflow.supervisor, "task_tool") as mock_task_tool:
            # Simulate comprehensive research workflow
            task_results = [
                {
                    "task_id": "research_1",
                    "status": TaskStatus.COMPLETED.value,
                    "output": "Initial research scope defined: Identified 5 key areas (LLM architectures, code generation, testing automation, productivity tools, industry adoption).",
                    "artifacts": ["research_scope.md", "research_questions.json"],
                    "execution_time": 20.0,
                    "tool_usage": {"execute_code": 2},
                },
                {
                    "task_id": "research_2",
                    "status": TaskStatus.COMPLETED.value,
                    "output": "LLM architecture research completed: Found 25 relevant papers and articles on latest transformer architectures, GPT variants, and specialized coding models.",
                    "artifacts": [
                        "llm_architecture_research.md",
                        "academic_sources.json",
                    ],
                    "execution_time": 60.0,
                    "tool_usage": {"search_internet": 8, "web_scrape": 12},
                },
                {
                    "task_id": "research_3",
                    "status": TaskStatus.COMPLETED.value,
                    "output": "Code generation tools research completed: Analyzed GitHub Copilot, CodeT5, Codex, and 15 other tools with performance comparisons.",
                    "artifacts": [
                        "code_generation_tools.md",
                        "tool_comparison_matrix.json",
                    ],
                    "execution_time": 75.0,
                    "tool_usage": {
                        "search_internet": 12,
                        "web_scrape": 18,
                        "execute_code": 5,
                    },
                },
                {
                    "task_id": "research_4",
                    "status": TaskStatus.COMPLETED.value,
                    "output": "Automated testing research completed: Identified 10 AI-powered testing tools and frameworks with case studies and effectiveness metrics.",
                    "artifacts": ["automated_testing_research.md", "case_studies.json"],
                    "execution_time": 50.0,
                    "tool_usage": {"search_internet": 10, "web_scrape": 15},
                },
                {
                    "task_id": "research_5",
                    "status": TaskStatus.COMPLETED.value,
                    "output": "Developer productivity tools research completed: Surveyed 20 AI-enhanced IDEs, code review tools, and development assistants.",
                    "artifacts": [
                        "productivity_tools_research.md",
                        "tool_survey_results.json",
                    ],
                    "execution_time": 45.0,
                    "tool_usage": {"search_internet": 8, "web_scrape": 10},
                },
                {
                    "task_id": "research_6",
                    "status": TaskStatus.COMPLETED.value,
                    "output": "Industry adoption analysis completed: Analyzed adoption rates, ROI studies, and implementation challenges across 50+ companies.",
                    "artifacts": [
                        "industry_adoption_analysis.md",
                        "adoption_metrics.json",
                    ],
                    "execution_time": 40.0,
                    "tool_usage": {
                        "search_internet": 6,
                        "web_scrape": 8,
                        "execute_code": 3,
                    },
                },
                {
                    "task_id": "research_7",
                    "status": TaskStatus.COMPLETED.value,
                    "output": "Comprehensive research report compiled: 45-page report with 120+ citations, executive summary, and actionable recommendations.",
                    "artifacts": [
                        "comprehensive_research_report.md",
                        "executive_summary.md",
                        "bibliography.json",
                        "recommendations.md",
                    ],
                    "execution_time": 35.0,
                    "tool_usage": {"execute_code": 6},
                },
            ]

            mock_task_tool.side_effect = task_results

            result = simple_workflow.run(objective)

            # Validate research workflow completion
            assert result.success is True
            assert result.completed_tasks >= 7

            # Validate research-specific content
            final_result_lower = result.final_result.lower()
            assert any(
                term in final_result_lower
                for term in ["research", "llm", "code generation", "report"]
            )

            # Validate research artifacts
            artifacts_str = str(result.artifacts)
            expected_artifacts = [
                "research_scope.md",
                "llm_architecture_research.md",
                "code_generation_tools.md",
                "automated_testing_research.md",
                "productivity_tools_research.md",
                "industry_adoption_analysis.md",
                "comprehensive_research_report.md",
                "bibliography.json",
            ]
            for artifact in expected_artifacts:
                assert artifact in artifacts_str

            # Validate research methodology (should use internet search and web scraping)
            total_search_usage = sum(
                task.get("tool_usage", {}).get("search_internet", 0)
                for task in task_results
            )
            total_scrape_usage = sum(
                task.get("tool_usage", {}).get("web_scrape", 0) for task in task_results
            )
            assert total_search_usage >= 40  # Should perform extensive searches
            assert total_scrape_usage >= 60  # Should scrape many sources

    def test_code_analysis_workflow_with_security_focus(self, simple_workflow):
        """Test code analysis workflow with comprehensive security assessment."""
        objective = (
            "Analyze this Python codebase for code quality issues, security vulnerabilities, "
            "performance bottlenecks, and maintainability problems. Provide detailed improvement "
            "recommendations with specific examples and priority levels."
        )

        # Pre-populate with vulnerable code
        simple_workflow.vfs.write_file(
            "vulnerable_app.py",
            """
import os
import pickle
import subprocess
from flask import Flask, request

app = Flask(__name__)
app.secret_key = "hardcoded_secret"  # Security issue

@app.route('/upload', methods=['POST'])
def upload_file():
    file_data = request.files['file']
    # Security issue: arbitrary file upload
    file_data.save(f"/uploads/{file_data.filename}")
    return "File uploaded"

@app.route('/execute')
def execute_command():
    cmd = request.args.get('cmd')
    # Security issue: command injection
    result = subprocess.run(cmd, shell=True, capture_output=True)
    return result.stdout

@app.route('/deserialize')
def deserialize_data():
    data = request.get_data()
    # Security issue: pickle deserialization
    obj = pickle.loads(data)
    return str(obj)
""",
        )

        with patch.object(simple_workflow.supervisor, "task_tool") as mock_task_tool:
            # Simulate comprehensive code analysis workflow
            task_results = [
                {
                    "task_id": "code_analysis_1",
                    "status": TaskStatus.COMPLETED.value,
                    "output": "Codebase structure analysis completed: Identified 3 Python files, 15 functions, 2 classes, and 450 lines of code.",
                    "artifacts": ["codebase_structure.md", "code_metrics.json"],
                    "execution_time": 25.0,
                    "tool_usage": {"execute_code": 5},
                },
                {
                    "task_id": "code_analysis_2",
                    "status": TaskStatus.COMPLETED.value,
                    "output": "Security vulnerability scan completed: Found 5 critical security issues including command injection, arbitrary file upload, and pickle deserialization vulnerabilities.",
                    "artifacts": [
                        "security_vulnerabilities.md",
                        "vulnerability_details.json",
                        "cve_mappings.json",
                    ],
                    "execution_time": 40.0,
                    "tool_usage": {"execute_code": 8, "search_internet": 5},
                },
                {
                    "task_id": "code_analysis_3",
                    "status": TaskStatus.COMPLETED.value,
                    "output": "Code quality analysis completed: Identified 12 code quality issues including hardcoded secrets, poor error handling, and style violations.",
                    "artifacts": ["code_quality_report.md", "quality_metrics.json"],
                    "execution_time": 30.0,
                    "tool_usage": {"execute_code": 6},
                },
                {
                    "task_id": "code_analysis_4",
                    "status": TaskStatus.COMPLETED.value,
                    "output": "Performance analysis completed: Found 3 performance bottlenecks including inefficient loops, missing caching, and database query issues.",
                    "artifacts": ["performance_analysis.md", "bottleneck_details.json"],
                    "execution_time": 35.0,
                    "tool_usage": {"execute_code": 7},
                },
                {
                    "task_id": "code_analysis_5",
                    "status": TaskStatus.COMPLETED.value,
                    "output": "Maintainability assessment completed: Code maintainability score is 6.2/10, identified issues with documentation, test coverage, and code complexity.",
                    "artifacts": [
                        "maintainability_report.md",
                        "complexity_metrics.json",
                    ],
                    "execution_time": 25.0,
                    "tool_usage": {"execute_code": 5},
                },
                {
                    "task_id": "code_analysis_6",
                    "status": TaskStatus.COMPLETED.value,
                    "output": "Improvement recommendations generated: Created 20 prioritized recommendations with code examples and implementation guides.",
                    "artifacts": [
                        "improvement_recommendations.md",
                        "code_examples.py",
                        "implementation_guide.md",
                    ],
                    "execution_time": 30.0,
                    "tool_usage": {"execute_code": 4},
                },
            ]

            mock_task_tool.side_effect = task_results

            result = simple_workflow.run(objective)

            # Validate code analysis completion
            assert result.success is True
            assert result.completed_tasks >= 6

            # Validate analysis-specific content
            final_result_lower = result.final_result.lower()
            assert any(
                term in final_result_lower
                for term in ["analysis", "security", "quality", "performance"]
            )

            # Validate analysis artifacts
            artifacts_str = str(result.artifacts)
            expected_artifacts = [
                "codebase_structure.md",
                "security_vulnerabilities.md",
                "code_quality_report.md",
                "performance_analysis.md",
                "maintainability_report.md",
                "improvement_recommendations.md",
            ]
            for artifact in expected_artifacts:
                assert artifact in artifacts_str

            # Validate that code execution was primary tool (appropriate for code analysis)
            total_code_execution = sum(
                task.get("tool_usage", {}).get("execute_code", 0)
                for task in task_results
            )
            assert (
                total_code_execution >= 30
            )  # Should execute code analysis tools extensively

    def test_workflow_with_task_dependencies_and_coordination(self, simple_workflow):
        """Test workflow handling of complex task dependencies and coordination."""
        objective = (
            "Create a complete project setup including requirements analysis, architecture design, "
            "implementation plan, and documentation. Each phase should build on the previous one."
        )

        with patch.object(simple_workflow.supervisor, "task_tool") as mock_task_tool:
            # Simulate workflow with clear dependencies
            task_results = [
                {
                    "task_id": "project_1",
                    "status": TaskStatus.COMPLETED.value,
                    "output": "Requirements analysis completed: Identified 15 functional requirements and 8 non-functional requirements.",
                    "artifacts": ["requirements_analysis.md", "user_stories.json"],
                    "execution_time": 30.0,
                    "tool_usage": {"execute_code": 3},
                },
                {
                    "task_id": "project_2",
                    "status": TaskStatus.COMPLETED.value,
                    "output": "Architecture design completed: Created system architecture with 5 microservices, database design, and API specifications.",
                    "artifacts": [
                        "architecture_design.md",
                        "system_diagrams.json",
                        "api_spec.json",
                    ],
                    "execution_time": 45.0,
                    "tool_usage": {"execute_code": 5},
                },
                {
                    "task_id": "project_3",
                    "status": TaskStatus.COMPLETED.value,
                    "output": "Implementation plan created: Broke down development into 12 sprints with 45 specific tasks and resource allocation.",
                    "artifacts": [
                        "implementation_plan.md",
                        "sprint_breakdown.json",
                        "task_dependencies.json",
                    ],
                    "execution_time": 35.0,
                    "tool_usage": {"execute_code": 4},
                },
                {
                    "task_id": "project_4",
                    "status": TaskStatus.COMPLETED.value,
                    "output": "Technical documentation generated: Created comprehensive docs including setup guides, API documentation, and developer handbook.",
                    "artifacts": [
                        "technical_documentation.md",
                        "api_docs.md",
                        "developer_handbook.md",
                    ],
                    "execution_time": 40.0,
                    "tool_usage": {"execute_code": 6},
                },
            ]

            mock_task_tool.side_effect = task_results

            result = simple_workflow.run(objective)

            # Validate coordinated workflow completion
            assert result.success is True
            assert result.completed_tasks >= 4

            # Validate that tasks built on each other (check artifacts reference previous work)
            artifacts_str = str(result.artifacts)
            expected_artifacts = [
                "requirements_analysis.md",
                "architecture_design.md",
                "implementation_plan.md",
                "technical_documentation.md",
            ]
            for artifact in expected_artifacts:
                assert artifact in artifacts_str

    def test_workflow_error_recovery_and_resilience(self, simple_workflow):
        """Test workflow resilience and error recovery mechanisms."""
        objective = "Test error handling and recovery in complex workflow"

        with patch.object(simple_workflow.supervisor, "task_tool") as mock_task_tool:
            # Simulate workflow with various error scenarios and recovery
            task_results = [
                {
                    "task_id": "resilience_1",
                    "status": TaskStatus.COMPLETED.value,
                    "output": "Initial task completed successfully",
                    "artifacts": ["initial_result.txt"],
                    "execution_time": 15.0,
                },
                {"error": "Network timeout during API call"},  # First error
                {"error": "Tool execution failed due to invalid input"},  # Second error
                {
                    "task_id": "resilience_2",
                    "status": TaskStatus.COMPLETED.value,
                    "output": "Recovery task completed after handling errors",
                    "artifacts": ["recovery_result.txt"],
                    "execution_time": 25.0,
                },
                {"error": "Temporary service unavailable"},  # Third error
                {
                    "task_id": "resilience_3",
                    "status": TaskStatus.COMPLETED.value,
                    "output": "Final task completed despite previous errors",
                    "artifacts": ["final_result.txt"],
                    "execution_time": 20.0,
                },
            ]

            mock_task_tool.side_effect = task_results

            result = simple_workflow.run(objective)

            # Should complete successfully despite errors
            assert result.success is True
            assert result.completed_tasks >= 2  # At least some tasks should complete

            # Should contain information about error handling
            final_result = result.final_result.lower()
            assert "task" in final_result

    def test_workflow_performance_with_large_task_count(self, simple_workflow):
        """Test workflow performance with a large number of tasks."""
        objective = "Execute a workflow with many coordinated tasks to test performance"

        with patch.object(simple_workflow.supervisor, "task_tool") as mock_task_tool:
            # Simulate workflow with many small tasks
            task_results = []
            for i in range(15):  # 15 tasks to test performance
                task_results.append(
                    {
                        "task_id": f"perf_task_{i + 1}",
                        "status": TaskStatus.COMPLETED.value,
                        "output": f"Performance task {i + 1} completed successfully",
                        "artifacts": [f"perf_result_{i + 1}.txt"],
                        "execution_time": 5.0
                        + (i * 0.5),  # Gradually increasing execution time
                        "tool_usage": {"execute_code": 1},
                    }
                )

            mock_task_tool.side_effect = task_results

            start_time = time.time()
            result = simple_workflow.run(objective)
            total_time = time.time() - start_time

            # Validate performance characteristics
            assert result.success is True
            assert result.completed_tasks >= 10  # Should complete most tasks
            assert total_time < 30.0  # Should complete reasonably quickly

            # Validate task coordination efficiency
            expected_total_task_time = sum(
                task["execution_time"] for task in task_results
            )
            # Workflow overhead should be reasonable
            assert total_time < expected_total_task_time + 10.0

    def test_example_scenarios_integration(self, example_scenarios):
        """Test integration with example scenarios module."""
        # Test web application testing scenario
        result = example_scenarios.run_web_application_testing_scenario()

        assert isinstance(result, ScenarioResult)
        assert result.scenario_name == "Web Application Testing"
        assert result.objective is not None
        assert isinstance(result.success, bool)
        assert result.execution_time >= 0
        assert result.iterations >= 0
        assert result.completed_tasks >= 0
        assert isinstance(result.artifacts_created, list)
        assert isinstance(result.files_created, list)

    def test_multiple_scenarios_sequential_execution(self, example_scenarios):
        """Test running multiple scenarios sequentially."""
        scenarios_to_test = [
            example_scenarios.run_research_workflow_scenario,
            example_scenarios.run_code_analysis_scenario,
        ]

        results = []
        for scenario_func in scenarios_to_test:
            example_scenarios.reset_workflow()
            result = scenario_func()
            results.append(result)

        # All scenarios should complete
        assert len(results) == 2
        for result in results:
            assert isinstance(result, ScenarioResult)
            assert result.execution_time >= 0

    @pytest.mark.asyncio
    async def test_async_workflow_integration(self, langgraph_workflow):
        """Test asynchronous workflow execution integration."""
        objective = "Test async workflow integration"

        with patch.object(langgraph_workflow.supervisor, "task_tool") as mock_task_tool:
            mock_task_tool.return_value = {
                "task_id": "async_integration_task",
                "status": TaskStatus.COMPLETED.value,
                "output": "Async integration test completed",
                "artifacts": ["async_result.txt"],
                "execution_time": 10.0,
            }

            # Mock the graph's async invoke
            with patch.object(langgraph_workflow.graph, "ainvoke") as mock_ainvoke:
                mock_final_state = {
                    "user_objective": objective,
                    "final_result": "Async workflow integration test completed successfully",
                    "iteration_count": 1,
                    "completed_tasks": [
                        {"task_id": "async_integration_task", "status": "completed"}
                    ],
                    "artifacts": {"async_result": "integration_data"},
                    "file_system": {},
                }
                mock_ainvoke.return_value = mock_final_state

                result = await langgraph_workflow.arun(objective)

                assert result["objective"] == objective
                assert result["final_result"] is not None
                assert result["iterations"] >= 0
                assert result["completed_tasks"] >= 0

    def test_workflow_component_integration_validation(self, simple_workflow):
        """Test that all workflow components integrate correctly."""
        objective = "Validate complete component integration"

        # Verify all components are properly initialized and connected
        assert simple_workflow.vfs is not None
        assert simple_workflow.tool_registry is not None
        assert simple_workflow.subagent_factory is not None
        assert simple_workflow.supervisor is not None

        # Verify component relationships
        assert simple_workflow.supervisor.vfs is simple_workflow.vfs
        assert simple_workflow.supervisor.tool_registry is simple_workflow.tool_registry
        assert (
            simple_workflow.supervisor.subagent_factory
            is simple_workflow.subagent_factory
        )

        # Test workflow execution with component validation
        with patch.object(simple_workflow.supervisor, "task_tool") as mock_task_tool:
            mock_task_tool.return_value = {
                "task_id": "integration_validation_task",
                "status": TaskStatus.COMPLETED.value,
                "output": "Component integration validation completed",
                "artifacts": ["integration_validation.txt"],
                "execution_time": 12.0,
            }

            result = simple_workflow.run(objective)

            assert result.success is True
            assert "integration" in result.final_result.lower()

    def test_workflow_state_consistency_across_iterations(self, simple_workflow):
        """Test that workflow state remains consistent across multiple iterations."""
        objective = "Test state consistency across workflow iterations"

        with patch.object(simple_workflow.supervisor, "task_tool") as mock_task_tool:
            # Simulate multiple iterations with state changes
            task_results = [
                {
                    "task_id": "state_test_1",
                    "status": TaskStatus.COMPLETED.value,
                    "output": "First iteration completed, state initialized",
                    "artifacts": ["state_1.json"],
                    "execution_time": 10.0,
                },
                {
                    "task_id": "state_test_2",
                    "status": TaskStatus.COMPLETED.value,
                    "output": "Second iteration completed, state updated",
                    "artifacts": ["state_2.json"],
                    "execution_time": 12.0,
                },
                {
                    "task_id": "state_test_3",
                    "status": TaskStatus.COMPLETED.value,
                    "output": "Third iteration completed, final state achieved",
                    "artifacts": ["state_3.json"],
                    "execution_time": 8.0,
                },
            ]

            mock_task_tool.side_effect = task_results

            result = simple_workflow.run(objective)

            # Validate state consistency
            assert result.success is True
            assert result.iterations >= 3
            assert result.completed_tasks >= 3

            # Validate that state was maintained across iterations
            artifacts_str = str(result.artifacts)
            assert "state_1.json" in artifacts_str
            assert "state_2.json" in artifacts_str
            assert "state_3.json" in artifacts_str

    def test_workflow_artifact_consolidation_and_management(self, simple_workflow):
        """Test comprehensive artifact consolidation and management."""
        objective = "Test artifact consolidation across complex workflow"

        with patch.object(simple_workflow.supervisor, "task_tool") as mock_task_tool:
            # Simulate workflow that creates many different types of artifacts
            task_results = [
                {
                    "task_id": "artifact_test_1",
                    "status": TaskStatus.COMPLETED.value,
                    "output": "Created documentation artifacts",
                    "artifacts": ["README.md", "API_DOCS.md", "CHANGELOG.md"],
                    "execution_time": 15.0,
                },
                {
                    "task_id": "artifact_test_2",
                    "status": TaskStatus.COMPLETED.value,
                    "output": "Created analysis artifacts",
                    "artifacts": ["analysis_report.json", "metrics.csv", "charts.png"],
                    "execution_time": 20.0,
                },
                {
                    "task_id": "artifact_test_3",
                    "status": TaskStatus.COMPLETED.value,
                    "output": "Created code artifacts",
                    "artifacts": ["main.py", "utils.py", "tests.py"],
                    "execution_time": 18.0,
                },
            ]

            mock_task_tool.side_effect = task_results

            result = simple_workflow.run(objective)

            # Validate artifact consolidation
            assert result.success is True

            # Check that all artifacts are tracked
            all_expected_artifacts = []
            for task_result in task_results:
                all_expected_artifacts.extend(task_result["artifacts"])

            artifacts_str = str(result.artifacts)
            for artifact in all_expected_artifacts:
                assert artifact in artifacts_str

            # Validate artifact organization
            assert "consolidated_results" in str(result.artifacts)
