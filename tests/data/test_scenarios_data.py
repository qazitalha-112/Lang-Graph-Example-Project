"""
Comprehensive test data and expected outcomes for workflow scenarios.

This module contains structured test data, expected outcomes, and validation
criteria for various workflow scenarios to ensure consistent and thorough testing.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from enum import Enum


class ScenarioType(Enum):
    """Types of test scenarios."""

    WEB_TESTING = "web_testing"
    RESEARCH = "research"
    CODE_ANALYSIS = "code_analysis"
    SECURITY_AUDIT = "security_audit"
    DATA_ANALYSIS = "data_analysis"
    DOCUMENTATION = "documentation"
    PERFORMANCE = "performance"
    ERROR_HANDLING = "error_handling"


@dataclass
class ExpectedOutcome:
    """Expected outcome for a test scenario."""

    min_tasks: int
    max_tasks: int
    min_iterations: int
    max_iterations: int
    expected_artifacts: List[str]
    required_tools: List[str]
    success_criteria: List[str]
    performance_thresholds: Dict[str, float]
    content_validation: Dict[str, List[str]]  # Key: artifact, Value: required content


@dataclass
class TestScenarioData:
    """Complete test scenario data with inputs and expected outcomes."""

    scenario_id: str
    scenario_type: ScenarioType
    name: str
    description: str
    objective: str
    setup_data: Dict[str, Any]
    expected_outcome: ExpectedOutcome
    validation_rules: List[str]


class TestScenariosData:
    """Collection of comprehensive test scenarios with expected outcomes."""

    @staticmethod
    def get_web_application_testing_scenarios() -> List[TestScenarioData]:
        """Get web application testing scenarios with expected outcomes."""
        return [
            TestScenarioData(
                scenario_id="web_test_ecommerce_comprehensive",
                scenario_type=ScenarioType.WEB_TESTING,
                name="E-commerce Web Application Comprehensive Testing",
                description="Complete testing of e-commerce application including security, performance, and usability",
                objective=(
                    "Test my e-commerce web application for bugs, security vulnerabilities, "
                    "performance issues, and usability problems. The application has user "
                    "authentication, product catalog, shopping cart, and payment processing."
                ),
                setup_data={
                    "application_url": "https://test-ecommerce.example.com",
                    "test_credentials": {
                        "username": "test_user",
                        "password": "test_password",
                    },
                    "application_features": [
                        "user_authentication",
                        "product_catalog",
                        "shopping_cart",
                        "payment_processing",
                        "order_management",
                    ],
                },
                expected_outcome=ExpectedOutcome(
                    min_tasks=5,
                    max_tasks=12,
                    min_iterations=3,
                    max_iterations=8,
                    expected_artifacts=[
                        "app_structure_analysis.md",
                        "auth_test_results.md",
                        "catalog_test_results.md",
                        "cart_test_results.md",
                        "payment_test_results.md",
                        "usability_report.md",
                        "security_findings.json",
                        "performance_analysis.json",
                        "comprehensive_test_report.md",
                    ],
                    required_tools=["web_scrape", "execute_code"],
                    success_criteria=[
                        "All major application features tested",
                        "Security vulnerabilities identified and documented",
                        "Performance bottlenecks analyzed",
                        "Usability issues documented with recommendations",
                        "Comprehensive test report generated",
                    ],
                    performance_thresholds={
                        "max_execution_time": 300.0,  # 5 minutes
                        "min_tasks_per_minute": 1.0,
                        "max_memory_usage_mb": 100.0,
                    },
                    content_validation={
                        "comprehensive_test_report.md": [
                            "security",
                            "performance",
                            "usability",
                            "bugs",
                            "recommendations",
                        ],
                        "security_findings.json": [
                            "vulnerabilities",
                            "severity",
                            "impact",
                        ],
                        "performance_analysis.json": [
                            "response_time",
                            "throughput",
                            "bottlenecks",
                        ],
                    },
                ),
                validation_rules=[
                    "Must identify at least 3 different types of issues",
                    "Must provide actionable recommendations",
                    "Must include risk assessment for security findings",
                    "Must generate executive summary",
                ],
            ),
            TestScenarioData(
                scenario_id="web_test_api_focused",
                scenario_type=ScenarioType.WEB_TESTING,
                name="API-Focused Web Application Testing",
                description="Testing focused on REST API endpoints, authentication, and data validation",
                objective=(
                    "Test the REST API endpoints of my web application for functionality, "
                    "security, performance, and data validation. Focus on authentication, "
                    "CRUD operations, and error handling."
                ),
                setup_data={
                    "api_base_url": "https://api.example.com/v1",
                    "api_documentation": "openapi_spec.json",
                    "endpoints": [
                        "/auth/login",
                        "/auth/logout",
                        "/users",
                        "/products",
                        "/orders",
                    ],
                },
                expected_outcome=ExpectedOutcome(
                    min_tasks=4,
                    max_tasks=8,
                    min_iterations=2,
                    max_iterations=6,
                    expected_artifacts=[
                        "api_test_plan.md",
                        "endpoint_test_results.json",
                        "auth_security_analysis.md",
                        "data_validation_report.md",
                        "api_performance_metrics.json",
                    ],
                    required_tools=["web_scrape", "execute_code"],
                    success_criteria=[
                        "All API endpoints tested",
                        "Authentication mechanisms validated",
                        "Data validation rules verified",
                        "Performance benchmarks established",
                    ],
                    performance_thresholds={
                        "max_execution_time": 180.0,
                        "min_tasks_per_minute": 1.5,
                        "max_memory_usage_mb": 50.0,
                    },
                    content_validation={
                        "api_test_plan.md": ["endpoints", "test_cases", "validation"],
                        "endpoint_test_results.json": [
                            "status_codes",
                            "response_times",
                            "errors",
                        ],
                    },
                ),
                validation_rules=[
                    "Must test all CRUD operations",
                    "Must validate authentication flows",
                    "Must check error handling",
                ],
            ),
        ]

    @staticmethod
    def get_research_workflow_scenarios() -> List[TestScenarioData]:
        """Get research workflow scenarios with expected outcomes."""
        return [
            TestScenarioData(
                scenario_id="research_ai_comprehensive",
                scenario_type=ScenarioType.RESEARCH,
                name="Comprehensive AI Research Workflow",
                description="In-depth research on AI developments with citation management",
                objective=(
                    "Research the latest developments in artificial intelligence and machine learning "
                    "for software development, including code generation, automated testing, and "
                    "developer productivity tools. Create a comprehensive report with proper citations."
                ),
                setup_data={
                    "research_domains": [
                        "large_language_models",
                        "code_generation",
                        "automated_testing",
                        "developer_productivity",
                    ],
                    "source_types": [
                        "academic_papers",
                        "industry_reports",
                        "case_studies",
                    ],
                    "time_range": "2023-2024",
                },
                expected_outcome=ExpectedOutcome(
                    min_tasks=6,
                    max_tasks=15,
                    min_iterations=4,
                    max_iterations=10,
                    expected_artifacts=[
                        "research_scope.md",
                        "literature_review.md",
                        "technology_analysis.md",
                        "case_studies.md",
                        "industry_trends.md",
                        "comprehensive_research_report.md",
                        "bibliography.json",
                        "executive_summary.md",
                    ],
                    required_tools=["search_internet", "web_scrape", "execute_code"],
                    success_criteria=[
                        "Comprehensive coverage of research domains",
                        "Proper citation management",
                        "Analysis of trends and patterns",
                        "Actionable insights and recommendations",
                    ],
                    performance_thresholds={
                        "max_execution_time": 400.0,
                        "min_sources_found": 20,
                        "min_tasks_per_minute": 0.8,
                    },
                    content_validation={
                        "comprehensive_research_report.md": [
                            "methodology",
                            "findings",
                            "analysis",
                            "conclusions",
                            "references",
                        ],
                        "bibliography.json": ["sources", "citations", "urls"],
                        "executive_summary.md": ["key_findings", "recommendations"],
                    },
                ),
                validation_rules=[
                    "Must include at least 20 credible sources",
                    "Must provide quantitative analysis where possible",
                    "Must identify future research directions",
                ],
            ),
            TestScenarioData(
                scenario_id="research_market_analysis",
                scenario_type=ScenarioType.RESEARCH,
                name="Market Analysis Research",
                description="Market research and competitive analysis workflow",
                objective=(
                    "Conduct market research and competitive analysis for AI-powered developer tools. "
                    "Analyze market size, key players, pricing strategies, and growth opportunities."
                ),
                setup_data={
                    "market_segment": "ai_developer_tools",
                    "competitors": ["GitHub Copilot", "Tabnine", "Kite", "CodeT5"],
                    "analysis_dimensions": [
                        "market_size",
                        "pricing",
                        "features",
                        "adoption",
                    ],
                },
                expected_outcome=ExpectedOutcome(
                    min_tasks=4,
                    max_tasks=10,
                    min_iterations=3,
                    max_iterations=7,
                    expected_artifacts=[
                        "market_overview.md",
                        "competitive_analysis.md",
                        "pricing_analysis.json",
                        "market_opportunities.md",
                        "swot_analysis.md",
                    ],
                    required_tools=["search_internet", "web_scrape", "execute_code"],
                    success_criteria=[
                        "Market size estimation",
                        "Competitive landscape mapping",
                        "Pricing strategy analysis",
                        "Growth opportunity identification",
                    ],
                    performance_thresholds={
                        "max_execution_time": 250.0,
                        "min_competitors_analyzed": 5,
                        "min_tasks_per_minute": 1.0,
                    },
                    content_validation={
                        "competitive_analysis.md": [
                            "competitors",
                            "features",
                            "strengths",
                            "weaknesses",
                        ],
                        "market_opportunities.md": [
                            "opportunities",
                            "threats",
                            "recommendations",
                        ],
                    },
                ),
                validation_rules=[
                    "Must analyze at least 5 competitors",
                    "Must provide market size estimates",
                    "Must identify specific opportunities",
                ],
            ),
        ]

    @staticmethod
    def get_code_analysis_scenarios() -> List[TestScenarioData]:
        """Get code analysis scenarios with expected outcomes."""
        return [
            TestScenarioData(
                scenario_id="code_analysis_security_focused",
                scenario_type=ScenarioType.CODE_ANALYSIS,
                name="Security-Focused Code Analysis",
                description="Comprehensive code analysis with emphasis on security vulnerabilities",
                objective=(
                    "Analyze this Python codebase for code quality issues, security vulnerabilities, "
                    "performance bottlenecks, and maintainability problems. Provide detailed "
                    "improvement recommendations with specific examples and priority levels."
                ),
                setup_data={
                    "codebase_files": {
                        "main.py": "vulnerable_web_app_code",
                        "utils.py": "utility_functions_with_issues",
                        "config.py": "configuration_with_secrets",
                    },
                    "analysis_focus": ["security", "performance", "maintainability"],
                    "frameworks": ["flask", "django"],
                },
                expected_outcome=ExpectedOutcome(
                    min_tasks=5,
                    max_tasks=12,
                    min_iterations=3,
                    max_iterations=8,
                    expected_artifacts=[
                        "codebase_structure.md",
                        "security_vulnerabilities.md",
                        "code_quality_report.md",
                        "performance_analysis.md",
                        "maintainability_report.md",
                        "improvement_recommendations.md",
                        "vulnerability_details.json",
                        "refactoring_suggestions.py",
                    ],
                    required_tools=["execute_code", "search_internet"],
                    success_criteria=[
                        "All security vulnerabilities identified",
                        "Performance bottlenecks documented",
                        "Code quality metrics calculated",
                        "Prioritized improvement recommendations",
                    ],
                    performance_thresholds={
                        "max_execution_time": 200.0,
                        "min_issues_found": 10,
                        "min_tasks_per_minute": 1.2,
                    },
                    content_validation={
                        "security_vulnerabilities.md": [
                            "sql_injection",
                            "xss",
                            "csrf",
                            "authentication",
                            "authorization",
                        ],
                        "improvement_recommendations.md": [
                            "priority",
                            "impact",
                            "effort",
                            "examples",
                        ],
                    },
                ),
                validation_rules=[
                    "Must identify critical security vulnerabilities",
                    "Must provide code examples for fixes",
                    "Must prioritize recommendations by impact",
                ],
            )
        ]

    @staticmethod
    def get_performance_test_scenarios() -> List[TestScenarioData]:
        """Get performance testing scenarios with expected outcomes."""
        return [
            TestScenarioData(
                scenario_id="performance_concurrent_load",
                scenario_type=ScenarioType.PERFORMANCE,
                name="Concurrent Load Performance Testing",
                description="Test system performance under concurrent load",
                objective="Execute multiple concurrent tasks to test system performance and scalability",
                setup_data={
                    "concurrent_tasks": 10,
                    "task_complexity": "medium",
                    "load_duration": 60,
                },
                expected_outcome=ExpectedOutcome(
                    min_tasks=8,
                    max_tasks=15,
                    min_iterations=2,
                    max_iterations=5,
                    expected_artifacts=[
                        "performance_metrics.json",
                        "load_test_results.md",
                        "scalability_analysis.md",
                    ],
                    required_tools=["execute_code"],
                    success_criteria=[
                        "All concurrent tasks complete successfully",
                        "Performance metrics within acceptable ranges",
                        "No memory leaks or resource exhaustion",
                    ],
                    performance_thresholds={
                        "max_execution_time": 120.0,
                        "min_throughput_tasks_per_second": 0.5,
                        "max_memory_usage_mb": 200.0,
                    },
                    content_validation={
                        "performance_metrics.json": [
                            "throughput",
                            "latency",
                            "memory_usage",
                            "cpu_usage",
                        ]
                    },
                ),
                validation_rules=[
                    "Must complete within time limits",
                    "Must maintain performance under load",
                    "Must not exceed memory thresholds",
                ],
            )
        ]

    @staticmethod
    def get_error_handling_scenarios() -> List[TestScenarioData]:
        """Get error handling scenarios with expected outcomes."""
        return [
            TestScenarioData(
                scenario_id="error_handling_resilience",
                scenario_type=ScenarioType.ERROR_HANDLING,
                name="Error Handling and Resilience Testing",
                description="Test system resilience and error recovery mechanisms",
                objective="Test error handling by simulating various failure scenarios and validating recovery",
                setup_data={
                    "error_types": [
                        "network_timeout",
                        "tool_failure",
                        "resource_unavailable",
                    ],
                    "recovery_strategies": [
                        "retry",
                        "fallback",
                        "graceful_degradation",
                    ],
                },
                expected_outcome=ExpectedOutcome(
                    min_tasks=3,
                    max_tasks=8,
                    min_iterations=2,
                    max_iterations=6,
                    expected_artifacts=[
                        "error_handling_report.md",
                        "recovery_analysis.json",
                        "resilience_metrics.md",
                    ],
                    required_tools=["execute_code"],
                    success_criteria=[
                        "System recovers from all error types",
                        "No data corruption during errors",
                        "Graceful degradation when appropriate",
                    ],
                    performance_thresholds={
                        "max_execution_time": 180.0,
                        "min_recovery_success_rate": 0.8,
                        "max_error_propagation_time": 30.0,
                    },
                    content_validation={
                        "error_handling_report.md": [
                            "error_types",
                            "recovery_strategies",
                            "success_rates",
                        ]
                    },
                ),
                validation_rules=[
                    "Must recover from at least 80% of errors",
                    "Must not lose data during recovery",
                    "Must complete within time limits despite errors",
                ],
            )
        ]

    @staticmethod
    def get_all_test_scenarios() -> List[TestScenarioData]:
        """Get all test scenarios with expected outcomes."""
        all_scenarios = []
        all_scenarios.extend(TestScenariosData.get_web_application_testing_scenarios())
        all_scenarios.extend(TestScenariosData.get_research_workflow_scenarios())
        all_scenarios.extend(TestScenariosData.get_code_analysis_scenarios())
        all_scenarios.extend(TestScenariosData.get_performance_test_scenarios())
        all_scenarios.extend(TestScenariosData.get_error_handling_scenarios())
        return all_scenarios

    @staticmethod
    def get_scenario_by_id(scenario_id: str) -> Optional[TestScenarioData]:
        """Get a specific test scenario by ID."""
        all_scenarios = TestScenariosData.get_all_test_scenarios()
        for scenario in all_scenarios:
            if scenario.scenario_id == scenario_id:
                return scenario
        return None

    @staticmethod
    def get_scenarios_by_type(scenario_type: ScenarioType) -> List[TestScenarioData]:
        """Get all test scenarios of a specific type."""
        all_scenarios = TestScenariosData.get_all_test_scenarios()
        return [s for s in all_scenarios if s.scenario_type == scenario_type]

    @staticmethod
    def validate_scenario_outcome(
        scenario_data: TestScenarioData, actual_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate actual workflow result against expected outcome.

        Args:
            scenario_data: Test scenario with expected outcomes
            actual_result: Actual workflow execution result

        Returns:
            Validation results with pass/fail status and details
        """
        validation_results = {
            "scenario_id": scenario_data.scenario_id,
            "overall_success": True,
            "validations": {},
            "failures": [],
            "warnings": [],
        }

        expected = scenario_data.expected_outcome

        # Validate task count
        actual_tasks = actual_result.get("completed_tasks", 0)
        if not (expected.min_tasks <= actual_tasks <= expected.max_tasks):
            validation_results["failures"].append(
                f"Task count {actual_tasks} not in expected range [{expected.min_tasks}, {expected.max_tasks}]"
            )
            validation_results["overall_success"] = False

        # Validate iteration count
        actual_iterations = actual_result.get("iterations", 0)
        if not (
            expected.min_iterations <= actual_iterations <= expected.max_iterations
        ):
            validation_results["warnings"].append(
                f"Iteration count {actual_iterations} not in expected range [{expected.min_iterations}, {expected.max_iterations}]"
            )

        # Validate artifacts
        actual_artifacts = actual_result.get("artifacts", {})
        artifacts_str = str(actual_artifacts)
        missing_artifacts = []
        for expected_artifact in expected.expected_artifacts:
            if expected_artifact not in artifacts_str:
                missing_artifacts.append(expected_artifact)

        if missing_artifacts:
            validation_results["failures"].append(
                f"Missing expected artifacts: {missing_artifacts}"
            )
            validation_results["overall_success"] = False

        # Validate performance thresholds
        execution_time = actual_result.get("execution_time", 0)
        if execution_time > expected.performance_thresholds.get(
            "max_execution_time", float("inf")
        ):
            validation_results["failures"].append(
                f"Execution time {execution_time}s exceeds threshold {expected.performance_thresholds['max_execution_time']}s"
            )
            validation_results["overall_success"] = False

        # Validate content (if final_result is available)
        final_result = actual_result.get("final_result", "")
        for artifact, required_content in expected.content_validation.items():
            if artifact in artifacts_str:
                missing_content = []
                for content in required_content:
                    if (
                        content.lower() not in final_result.lower()
                        and content.lower() not in artifacts_str.lower()
                    ):
                        missing_content.append(content)

                if missing_content:
                    validation_results["warnings"].append(
                        f"Artifact {artifact} missing expected content: {missing_content}"
                    )

        # Validate success criteria
        success_criteria_met = 0
        for criteria in expected.success_criteria:
            # Simple keyword-based validation
            criteria_keywords = criteria.lower().split()
            if any(
                keyword in final_result.lower() or keyword in artifacts_str.lower()
                for keyword in criteria_keywords
            ):
                success_criteria_met += 1

        success_rate = success_criteria_met / len(expected.success_criteria)
        validation_results["validations"]["success_criteria_rate"] = success_rate

        if success_rate < 0.7:  # At least 70% of success criteria should be met
            validation_results["failures"].append(
                f"Only {success_criteria_met}/{len(expected.success_criteria)} success criteria met"
            )
            validation_results["overall_success"] = False

        return validation_results


# Sample test data for different scenarios
SAMPLE_VULNERABLE_CODE = """
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
"""

SAMPLE_API_SPECIFICATION = """{
    "openapi": "3.0.0",
    "info": {
        "title": "E-commerce API",
        "version": "1.0.0"
    },
    "paths": {
        "/api/products": {
            "get": {
                "summary": "List products",
                "responses": {
                    "200": {"description": "List of products"}
                }
            }
        },
        "/api/cart": {
            "post": {
                "summary": "Add item to cart",
                "responses": {
                    "201": {"description": "Item added"}
                }
            }
        }
    }
}"""

SAMPLE_RESEARCH_SOURCES = [
    {
        "title": "Large Language Models for Code Generation",
        "authors": ["Smith, J.", "Doe, A."],
        "year": 2024,
        "url": "https://example.com/paper1",
        "type": "academic_paper",
    },
    {
        "title": "GitHub Copilot: AI-Powered Code Completion",
        "authors": ["GitHub Team"],
        "year": 2023,
        "url": "https://github.com/features/copilot",
        "type": "industry_report",
    },
]
