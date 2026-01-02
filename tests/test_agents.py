import pytest
from unittest.mock import Mock, MagicMock
from agents import ResearchAgent, AnalysisAgent, SummaryAgent


class TestResearchAgent:
    """Test suite for ResearchAgent class."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM for testing."""
        llm = Mock()
        llm.invoke = Mock(return_value="Mocked research response")
        return llm

    @pytest.fixture
    def research_agent(self, mock_llm):
        """Create a ResearchAgent instance with mocked LLM."""
        return ResearchAgent(mock_llm)

    def test_init(self, mock_llm):
        """Test ResearchAgent initialization."""
        agent = ResearchAgent(mock_llm)
        assert agent.llm == mock_llm

    def test_research_topic_basic(self, research_agent, mock_llm):
        """Test basic research_topic functionality."""
        topic = "artificial intelligence"
        result = research_agent.research_topic(topic)

        # Verify the LLM was called
        mock_llm.invoke.assert_called_once()

        # Verify the result
        assert result == "Mocked research response"

    def test_research_topic_prompt_contains_topic(self, research_agent, mock_llm):
        """Test that the prompt contains the topic."""
        topic = "machine learning"
        research_agent.research_topic(topic)

        # Get the prompt that was passed to invoke
        call_args = mock_llm.invoke.call_args[0][0]
        assert topic in call_args
        assert "research agent" in call_args.lower()

    def test_research_topic_with_special_characters(self, research_agent, mock_llm):
        """Test research with special characters in topic."""
        topic = "AI & ML: Future & Trends"
        result = research_agent.research_topic(topic)

        assert result == "Mocked research response"
        mock_llm.invoke.assert_called_once()

    def test_research_topic_empty_string(self, research_agent, mock_llm):
        """Test research with empty topic string."""
        topic = ""
        result = research_agent.research_topic(topic)

        assert result == "Mocked research response"
        mock_llm.invoke.assert_called_once()


class TestAnalysisAgent:
    """Test suite for AnalysisAgent class."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM for testing."""
        llm = Mock()
        llm.invoke = Mock(return_value="Mocked analysis response")
        return llm

    @pytest.fixture
    def analysis_agent(self, mock_llm):
        """Create an AnalysisAgent instance with mocked LLM."""
        return AnalysisAgent(mock_llm)

    def test_init(self, mock_llm):
        """Test AnalysisAgent initialization."""
        agent = AnalysisAgent(mock_llm)
        assert agent.llm == mock_llm

    def test_analyze_research_basic(self, analysis_agent, mock_llm):
        """Test basic analyze_research functionality."""
        research_data = "This is research data about AI."
        topic = "artificial intelligence"

        result = analysis_agent.analyze_research(research_data, topic)

        # Verify the LLM was called
        mock_llm.invoke.assert_called_once()

        # Verify the result
        assert result == "Mocked analysis response"

    def test_analyze_research_prompt_contains_data_and_topic(self, analysis_agent, mock_llm):
        """Test that the prompt contains both research data and topic."""
        research_data = "Key findings about neural networks."
        topic = "deep learning"

        analysis_agent.analyze_research(research_data, topic)

        # Get the prompt that was passed to invoke
        call_args = mock_llm.invoke.call_args[0][0]
        assert research_data in call_args
        assert topic in call_args
        assert "analysis agent" in call_args.lower()

    def test_analyze_research_with_long_data(self, analysis_agent, mock_llm):
        """Test analysis with long research data."""
        research_data = "A" * 10000  # Very long research data
        topic = "test topic"

        result = analysis_agent.analyze_research(research_data, topic)

        assert result == "Mocked analysis response"
        mock_llm.invoke.assert_called_once()

    def test_analyze_research_empty_data(self, analysis_agent, mock_llm):
        """Test analysis with empty research data."""
        research_data = ""
        topic = "test topic"

        result = analysis_agent.analyze_research(research_data, topic)

        assert result == "Mocked analysis response"
        mock_llm.invoke.assert_called_once()


class TestSummaryAgent:
    """Test suite for SummaryAgent class."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM for testing."""
        llm = Mock()
        llm.invoke = Mock(return_value="Mocked summary response")
        return llm

    @pytest.fixture
    def summary_agent(self, mock_llm):
        """Create a SummaryAgent instance with mocked LLM."""
        return SummaryAgent(mock_llm)

    def test_init(self, mock_llm):
        """Test SummaryAgent initialization."""
        agent = SummaryAgent(mock_llm)
        assert agent.llm == mock_llm

    def test_create_summary_basic(self, summary_agent, mock_llm):
        """Test basic create_summary functionality."""
        research_data = "Research findings about AI."
        analysis = "Analysis of AI trends."
        topic = "artificial intelligence"

        result = summary_agent.create_summary(research_data, analysis, topic)

        # Verify the LLM was called
        mock_llm.invoke.assert_called_once()

        # Verify the result
        assert result == "Mocked summary response"

    def test_create_summary_prompt_contains_all_inputs(self, summary_agent, mock_llm):
        """Test that the prompt contains research data, analysis, and topic."""
        research_data = "Research about neural networks."
        analysis = "Analysis of deep learning."
        topic = "machine learning"

        summary_agent.create_summary(research_data, analysis, topic)

        # Get the prompt that was passed to invoke
        call_args = mock_llm.invoke.call_args[0][0]
        assert research_data in call_args
        assert analysis in call_args
        assert topic in call_args
        assert "summary agent" in call_args.lower()

    def test_create_summary_with_multiline_data(self, summary_agent, mock_llm):
        """Test summary creation with multiline data."""
        research_data = "Line 1\nLine 2\nLine 3"
        analysis = "Analysis line 1\nAnalysis line 2"
        topic = "test topic"

        result = summary_agent.create_summary(research_data, analysis, topic)

        assert result == "Mocked summary response"
        mock_llm.invoke.assert_called_once()

    def test_create_summary_with_special_characters(self, summary_agent, mock_llm):
        """Test summary with special characters."""
        research_data = "Research with $pecial ch@racters!"
        analysis = "Analysis with émojis 🚀"
        topic = "special & unusual"

        result = summary_agent.create_summary(research_data, analysis, topic)

        assert result == "Mocked summary response"
        mock_llm.invoke.assert_called_once()

    def test_create_summary_empty_inputs(self, summary_agent, mock_llm):
        """Test summary with empty inputs."""
        result = summary_agent.create_summary("", "", "")

        assert result == "Mocked summary response"
        mock_llm.invoke.assert_called_once()


class TestAgentsIntegration:
    """Integration tests for all agents working together."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM for testing."""
        llm = Mock()
        return llm

    def test_all_agents_use_same_llm_instance(self, mock_llm):
        """Test that all agents can use the same LLM instance."""
        research_agent = ResearchAgent(mock_llm)
        analysis_agent = AnalysisAgent(mock_llm)
        summary_agent = SummaryAgent(mock_llm)

        assert research_agent.llm is mock_llm
        assert analysis_agent.llm is mock_llm
        assert summary_agent.llm is mock_llm

    def test_sequential_agent_calls(self, mock_llm):
        """Test sequential calls through all agents."""
        # Set up different responses for each call
        mock_llm.invoke = Mock(side_effect=[
            "Research result",
            "Analysis result",
            "Summary result"
        ])

        research_agent = ResearchAgent(mock_llm)
        analysis_agent = AnalysisAgent(mock_llm)
        summary_agent = SummaryAgent(mock_llm)

        topic = "test topic"

        # Simulate the pipeline
        research = research_agent.research_topic(topic)
        assert research == "Research result"

        analysis = analysis_agent.analyze_research(research, topic)
        assert analysis == "Analysis result"

        summary = summary_agent.create_summary(research, analysis, topic)
        assert summary == "Summary result"

        # Verify all three calls were made
        assert mock_llm.invoke.call_count == 3
