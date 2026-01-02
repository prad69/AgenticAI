import pytest
import os
from unittest.mock import Mock, MagicMock, patch, mock_open
from orchestrator import MultiAgentOrchestrator


class TestMultiAgentOrchestrator:
    """Test suite for MultiAgentOrchestrator class."""

    @pytest.fixture
    def mock_env(self):
        """Mock environment variables."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-api-key'}):
            yield

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        llm = Mock()
        llm.invoke = Mock(return_value="Mocked LLM response")
        return llm

    @pytest.fixture
    def orchestrator(self, mock_env, mock_llm):
        """Create an orchestrator instance with mocked dependencies."""
        with patch('orchestrator.ChatOpenAI', return_value=mock_llm):
            return MultiAgentOrchestrator()

    def test_init_creates_agents(self, mock_env, mock_llm):
        """Test that initialization creates all necessary agents."""
        with patch('orchestrator.ChatOpenAI', return_value=mock_llm):
            orchestrator = MultiAgentOrchestrator()

            assert orchestrator.llm is not None
            assert orchestrator.research_agent is not None
            assert orchestrator.analysis_agent is not None
            assert orchestrator.summary_agent is not None

    def test_init_creates_llm_with_correct_params(self, mock_env):
        """Test that LLM is initialized with correct parameters."""
        with patch('orchestrator.ChatOpenAI') as mock_chat_openai:
            MultiAgentOrchestrator()

            # Verify ChatOpenAI was called with correct parameters
            mock_chat_openai.assert_called_once()
            call_kwargs = mock_chat_openai.call_args[1]
            assert call_kwargs['api_key'] == 'test-api-key'
            assert call_kwargs['temperature'] == 0.7
            assert call_kwargs['model'] == 'gpt-3.5-turbo'

    def test_run_research_pipeline_basic(self, orchestrator, capsys):
        """Test basic pipeline execution."""
        topic = "artificial intelligence"

        # Mock agent responses
        orchestrator.research_agent.research_topic = Mock(return_value="Research data")
        orchestrator.analysis_agent.analyze_research = Mock(return_value="Analysis data")
        orchestrator.summary_agent.create_summary = Mock(return_value="Summary data")

        result = orchestrator.run_research_pipeline(topic)

        # Verify all agents were called
        orchestrator.research_agent.research_topic.assert_called_once_with(topic)
        orchestrator.analysis_agent.analyze_research.assert_called_once_with("Research data", topic)
        orchestrator.summary_agent.create_summary.assert_called_once_with(
            "Research data", "Analysis data", topic
        )

        # Verify result structure
        assert result['topic'] == topic
        assert result['research_data'] == "Research data"
        assert result['analysis'] == "Analysis data"
        assert result['final_summary'] == "Summary data"

        # Verify console output
        captured = capsys.readouterr()
        assert "Starting research pipeline" in captured.out
        assert "Research Agent working" in captured.out
        assert "Analysis Agent working" in captured.out
        assert "Summary Agent working" in captured.out
        assert "Pipeline completed" in captured.out

    def test_run_research_pipeline_returns_correct_dict(self, orchestrator):
        """Test that pipeline returns a dictionary with all required keys."""
        topic = "machine learning"

        orchestrator.research_agent.research_topic = Mock(return_value="R")
        orchestrator.analysis_agent.analyze_research = Mock(return_value="A")
        orchestrator.summary_agent.create_summary = Mock(return_value="S")

        result = orchestrator.run_research_pipeline(topic)

        # Verify all required keys are present
        assert 'topic' in result
        assert 'research_data' in result
        assert 'analysis' in result
        assert 'final_summary' in result

        # Verify correct values
        assert result['topic'] == topic
        assert result['research_data'] == "R"
        assert result['analysis'] == "A"
        assert result['final_summary'] == "S"

    def test_run_research_pipeline_with_special_characters(self, orchestrator):
        """Test pipeline with special characters in topic."""
        topic = "AI & ML: 2024 Trends!"

        orchestrator.research_agent.research_topic = Mock(return_value="Research")
        orchestrator.analysis_agent.analyze_research = Mock(return_value="Analysis")
        orchestrator.summary_agent.create_summary = Mock(return_value="Summary")

        result = orchestrator.run_research_pipeline(topic)

        assert result['topic'] == topic

    def test_run_research_pipeline_agents_called_in_sequence(self, orchestrator):
        """Test that agents are called in the correct sequence."""
        topic = "test"
        call_order = []

        def research_side_effect(t):
            call_order.append('research')
            return "Research"

        def analysis_side_effect(r, t):
            call_order.append('analysis')
            return "Analysis"

        def summary_side_effect(r, a, t):
            call_order.append('summary')
            return "Summary"

        orchestrator.research_agent.research_topic = Mock(side_effect=research_side_effect)
        orchestrator.analysis_agent.analyze_research = Mock(side_effect=analysis_side_effect)
        orchestrator.summary_agent.create_summary = Mock(side_effect=summary_side_effect)

        orchestrator.run_research_pipeline(topic)

        assert call_order == ['research', 'analysis', 'summary']

    def test_save_results_with_default_filename(self, orchestrator):
        """Test saving results with default filename."""
        results = {
            'topic': 'artificial intelligence',
            'research_data': 'Research content',
            'analysis': 'Analysis content',
            'final_summary': 'Summary content'
        }

        expected_filename = "research_report_artificial_intelligence.txt"
        expected_content = """RESEARCH REPORT: artificial intelligence
==================================================

RESEARCH DATA:
Research content

==================================================

ANALYSIS:
Analysis content

==================================================

FINAL SUMMARY:
Summary content"""

        with patch('builtins.open', mock_open()) as mocked_file:
            orchestrator.save_results(results)

            # Verify file was opened with correct filename
            mocked_file.assert_called_once_with(expected_filename, 'w')

            # Get all write calls and combine them
            handle = mocked_file()
            write_calls = [call[0][0] for call in handle.write.call_args_list]
            actual_content = ''.join(write_calls)

            assert actual_content == expected_content

    def test_save_results_with_custom_filename(self, orchestrator):
        """Test saving results with custom filename."""
        results = {
            'topic': 'test topic',
            'research_data': 'R',
            'analysis': 'A',
            'final_summary': 'S'
        }

        custom_filename = "custom_report.txt"

        with patch('builtins.open', mock_open()) as mocked_file:
            orchestrator.save_results(results, filename=custom_filename)

            mocked_file.assert_called_once_with(custom_filename, 'w')

    def test_save_results_replaces_spaces_in_topic(self, orchestrator):
        """Test that spaces in topic are replaced with underscores in filename."""
        results = {
            'topic': 'machine learning basics',
            'research_data': 'R',
            'analysis': 'A',
            'final_summary': 'S'
        }

        with patch('builtins.open', mock_open()) as mocked_file:
            orchestrator.save_results(results)

            # Verify filename has underscores instead of spaces
            expected_filename = "research_report_machine_learning_basics.txt"
            mocked_file.assert_called_once_with(expected_filename, 'w')

    def test_save_results_writes_correct_structure(self, orchestrator):
        """Test that saved file has correct structure."""
        results = {
            'topic': 'test',
            'research_data': 'RESEARCH',
            'analysis': 'ANALYSIS',
            'final_summary': 'SUMMARY'
        }

        with patch('builtins.open', mock_open()) as mocked_file:
            orchestrator.save_results(results)

            handle = mocked_file()
            write_calls = [call[0][0] for call in handle.write.call_args_list]
            full_content = ''.join(write_calls)

            # Verify structure
            assert 'RESEARCH REPORT: test' in full_content
            assert 'RESEARCH DATA:' in full_content
            assert 'RESEARCH' in full_content
            assert 'ANALYSIS:' in full_content
            assert 'ANALYSIS' in full_content
            assert 'FINAL SUMMARY:' in full_content
            assert 'SUMMARY' in full_content
            assert '=' * 50 in full_content

    def test_save_results_prints_confirmation(self, orchestrator, capsys):
        """Test that save_results prints confirmation message."""
        results = {
            'topic': 'test',
            'research_data': 'R',
            'analysis': 'A',
            'final_summary': 'S'
        }

        filename = "test_report.txt"

        with patch('builtins.open', mock_open()):
            orchestrator.save_results(results, filename=filename)

        captured = capsys.readouterr()
        assert f"Results saved to {filename}" in captured.out

    def test_save_results_with_multiline_content(self, orchestrator):
        """Test saving results with multiline content."""
        results = {
            'topic': 'test',
            'research_data': 'Line 1\nLine 2\nLine 3',
            'analysis': 'Analysis line 1\nAnalysis line 2',
            'final_summary': 'Summary\nwith\nmultiple\nlines'
        }

        with patch('builtins.open', mock_open()) as mocked_file:
            orchestrator.save_results(results)

            handle = mocked_file()
            write_calls = [call[0][0] for call in handle.write.call_args_list]
            full_content = ''.join(write_calls)

            assert 'Line 1\nLine 2\nLine 3' in full_content
            assert 'Analysis line 1\nAnalysis line 2' in full_content
            assert 'Summary\nwith\nmultiple\nlines' in full_content


class TestMultiAgentOrchestratorIntegration:
    """Integration tests for the full orchestrator."""

    @pytest.fixture
    def mock_env(self):
        """Mock environment variables."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-api-key'}):
            yield

    def test_full_pipeline_integration(self, mock_env):
        """Test the complete pipeline flow."""
        mock_llm = Mock()
        mock_llm.invoke = Mock(side_effect=[
            "Detailed research about AI",
            "Comprehensive analysis of AI",
            "Executive summary of AI research"
        ])

        with patch('orchestrator.ChatOpenAI', return_value=mock_llm):
            orchestrator = MultiAgentOrchestrator()
            result = orchestrator.run_research_pipeline("artificial intelligence")

            # Verify pipeline executed completely
            assert result['topic'] == "artificial intelligence"
            assert result['research_data'] == "Detailed research about AI"
            assert result['analysis'] == "Comprehensive analysis of AI"
            assert result['final_summary'] == "Executive summary of AI research"

            # Verify LLM was called three times
            assert mock_llm.invoke.call_count == 3

    def test_pipeline_and_save_integration(self, mock_env):
        """Test running pipeline and saving results."""
        mock_llm = Mock()
        mock_llm.invoke = Mock(return_value="Test response")

        with patch('orchestrator.ChatOpenAI', return_value=mock_llm):
            orchestrator = MultiAgentOrchestrator()
            result = orchestrator.run_research_pipeline("test topic")

            with patch('builtins.open', mock_open()) as mocked_file:
                orchestrator.save_results(result)

                # Verify file was created
                mocked_file.assert_called_once()

                # Verify content was written
                handle = mocked_file()
                assert handle.write.called

    def test_missing_api_key_environment_variable(self):
        """Test behavior when OPENAI_API_KEY is not set."""
        # This test verifies that the code attempts to get the API key
        # The actual behavior depends on how dotenv and OpenAI handle missing keys
        with patch.dict(os.environ, {}, clear=True):
            with patch('orchestrator.load_dotenv'):
                with patch('orchestrator.ChatOpenAI') as mock_chat:
                    MultiAgentOrchestrator()

                    # Verify ChatOpenAI was called (even with None api_key)
                    assert mock_chat.called
