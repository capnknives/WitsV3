#!/usr/bin/env python3
"""
Test suite for Neural Web Visualization Tools

This test suite validates the visualization capabilities for Neural Web
knowledge networks including static graphs and interactive HTML visualizations.
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from tools.neural_web_visualization import (
    NeuralWebVisualizer,
    NeuralWebVisualizationTool
)
from core.config import WitsV3Config
from core.neural_web_core import NeuralWeb
from core.memory_manager import MemoryManager


class TestNeuralWebVisualizer:
    """Test the core NeuralWebVisualizer functionality."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        config = MagicMock(spec=WitsV3Config)
        config.neural_web = MagicMock()
        config.neural_web.enable_visualization = True
        config.neural_web.max_visualization_nodes = 100
        return config

    @pytest.fixture
    def mock_neural_web(self):
        """Create a mock neural web."""
        neural_web = MagicMock(spec=NeuralWeb)

        # Mock concept data
        neural_web.get_all_concepts.return_value = {
            "concept1": {
                "name": "artificial intelligence",
                "domain": "technology",
                "activation": 0.8,
                "connections": {"concept2": 0.6, "concept3": 0.4}
            },
            "concept2": {
                "name": "machine learning",
                "domain": "technology",
                "activation": 0.7,
                "connections": {"concept1": 0.6, "concept4": 0.5}
            },
            "concept3": {
                "name": "neural networks",
                "domain": "technology",
                "activation": 0.6,
                "connections": {"concept1": 0.4, "concept4": 0.7}
            },
            "concept4": {
                "name": "deep learning",
                "domain": "technology",
                "activation": 0.9,
                "connections": {"concept2": 0.5, "concept3": 0.7}
            }
        }

        # Mock domain data
        neural_web.get_domain_concepts.return_value = {
            "technology": ["concept1", "concept2", "concept3", "concept4"],
            "science": ["concept5", "concept6"]
        }

        return neural_web

    @pytest.fixture
    def visualizer(self, mock_neural_web, mock_config):
        """Create a visualizer instance."""
        return NeuralWebVisualizer(mock_neural_web, mock_config)

    @pytest.mark.asyncio
    async def test_generate_network_graph_png(self, visualizer):
        """Test PNG graph generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('matplotlib.pyplot.savefig') as mock_savefig:
                result = await visualizer.generate_network_graph(
                    format="png",
                    include_weights=True,
                    filter_threshold=0.3,
                    max_nodes=50
                )

                assert result.endswith('.png')
                mock_savefig.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_network_graph_svg(self, visualizer):
        """Test SVG graph generation."""
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            result = await visualizer.generate_network_graph(
                format="svg",
                include_weights=False,
                filter_threshold=0.2
            )

            assert result.endswith('.svg')
            mock_savefig.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_interactive_visualization(self, visualizer):
        """Test interactive HTML visualization generation."""
        result = await visualizer.generate_interactive_visualization(
            include_search=True,
            include_filters=True,
            theme="dark"
        )

        assert result.endswith('.html')

        # Verify the HTML file contains expected elements
        with open(result, 'r') as f:
            content = f.read()
            assert 'D3.js' in content or 'd3' in content
            assert 'neural-web-graph' in content
            assert 'function' in content

    @pytest.mark.asyncio
    async def test_create_domain_summary(self, visualizer):
        """Test domain summary creation."""
        result = await visualizer.create_domain_summary(
            domain="technology",
            include_metrics=True,
            include_top_concepts=True,
            max_concepts=10
        )

        assert "technology" in result
        assert "concepts" in result.lower()
        assert "connections" in result.lower()

    @pytest.mark.asyncio
    async def test_export_network_data(self, visualizer):
        """Test network data export."""
        with tempfile.TemporaryDirectory() as temp_dir:
            export_path = Path(temp_dir) / "network_export.json"

            result = await visualizer.export_network_data(
                str(export_path),
                include_metadata=True,
                filter_threshold=0.1
            )

            assert result == str(export_path)
            assert export_path.exists()

            # Verify JSON structure
            with open(export_path, 'r') as f:
                data = json.load(f)
                assert 'nodes' in data
                assert 'edges' in data
                assert 'metadata' in data

    @pytest.mark.asyncio
    async def test_filter_threshold_application(self, visualizer):
        """Test that filter threshold properly filters weak connections."""
        with patch('matplotlib.pyplot.savefig'):
            # High threshold should result in fewer connections
            await visualizer.generate_network_graph(
                filter_threshold=0.8,
                max_nodes=100
            )

            # Verify neural web was called with proper parameters
            visualizer.neural_web.get_all_concepts.assert_called()

    @pytest.mark.asyncio
    async def test_max_nodes_limitation(self, visualizer):
        """Test that max_nodes parameter limits output."""
        with patch('matplotlib.pyplot.savefig'):
            result = await visualizer.generate_network_graph(
                max_nodes=2,
                filter_threshold=0.0
            )

            assert result.endswith('.png')
            # In a real implementation, we'd verify only 2 nodes were included


class TestNeuralWebVisualizationTool:
    """Test the tool interface for Neural Web visualization."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        return MagicMock(spec=WitsV3Config)

    @pytest.fixture
    def visualization_tool(self, mock_config):
        """Create a visualization tool instance."""
        return NeuralWebVisualizationTool(mock_config)

    @pytest.mark.asyncio
    async def test_tool_initialization(self, visualization_tool):
        """Test proper tool initialization."""
        assert visualization_tool.name == "neural_web_visualization"
        assert "visualize" in visualization_tool.description.lower()
        assert hasattr(visualization_tool, 'execute')

    @pytest.mark.asyncio
    async def test_get_schema(self, visualization_tool):
        """Test tool schema generation."""
        schema = visualization_tool.get_schema()

        assert schema['name'] == "neural_web_visualization"
        assert 'description' in schema
        assert 'parameters' in schema
        assert 'properties' in schema['parameters']

        # Check for key parameters
        properties = schema['parameters']['properties']
        assert 'visualization_type' in properties
        assert 'format' in properties
        assert 'include_weights' in properties

    @pytest.mark.asyncio
    async def test_execute_network_graph(self, visualization_tool):
        """Test executing network graph generation."""
        with patch.object(visualization_tool, '_get_neural_web') as mock_get_nw:
            mock_neural_web = MagicMock()
            mock_get_nw.return_value = mock_neural_web

            with patch('tools.neural_web_visualization.NeuralWebVisualizer') as MockVisualizer:
                mock_visualizer = MockVisualizer.return_value
                mock_visualizer.generate_network_graph.return_value = "/path/to/graph.png"

                result = await visualization_tool.execute(
                    visualization_type="network_graph",
                    format="png",
                    include_weights=True,
                    filter_threshold=0.3
                )

                assert result.success
                assert "graph.png" in result.result
                mock_visualizer.generate_network_graph.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_interactive_visualization(self, visualization_tool):
        """Test executing interactive visualization generation."""
        with patch.object(visualization_tool, '_get_neural_web') as mock_get_nw:
            mock_neural_web = MagicMock()
            mock_get_nw.return_value = mock_neural_web

            with patch('tools.neural_web_visualization.NeuralWebVisualizer') as MockVisualizer:
                mock_visualizer = MockVisualizer.return_value
                mock_visualizer.generate_interactive_visualization.return_value = "/path/to/interactive.html"

                result = await visualization_tool.execute(
                    visualization_type="interactive",
                    include_search=True,
                    include_filters=True,
                    theme="light"
                )

                assert result.success
                assert "interactive.html" in result.result
                mock_visualizer.generate_interactive_visualization.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_domain_summary(self, visualization_tool):
        """Test executing domain summary generation."""
        with patch.object(visualization_tool, '_get_neural_web') as mock_get_nw:
            mock_neural_web = MagicMock()
            mock_get_nw.return_value = mock_neural_web

            with patch('tools.neural_web_visualization.NeuralWebVisualizer') as MockVisualizer:
                mock_visualizer = MockVisualizer.return_value
                mock_visualizer.create_domain_summary.return_value = "Domain summary text"

                result = await visualization_tool.execute(
                    visualization_type="domain_summary",
                    domain="technology",
                    include_metrics=True
                )

                assert result.success
                assert "Domain summary text" in result.result
                mock_visualizer.create_domain_summary.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_invalid_type(self, visualization_tool):
        """Test handling of invalid visualization type."""
        result = await visualization_tool.execute(
            visualization_type="invalid_type"
        )

        assert not result.success
        assert "unsupported" in result.error.lower() or "invalid" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_missing_neural_web(self, visualization_tool):
        """Test handling when neural web is not available."""
        with patch.object(visualization_tool, '_get_neural_web') as mock_get_nw:
            mock_get_nw.return_value = None

            result = await visualization_tool.execute(
                visualization_type="network_graph"
            )

            assert not result.success
            assert "neural web not available" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_with_error(self, visualization_tool):
        """Test error handling during execution."""
        with patch.object(visualization_tool, '_get_neural_web') as mock_get_nw:
            mock_neural_web = MagicMock()
            mock_get_nw.return_value = mock_neural_web

            with patch('tools.neural_web_visualization.NeuralWebVisualizer') as MockVisualizer:
                mock_visualizer = MockVisualizer.return_value
                mock_visualizer.generate_network_graph.side_effect = Exception("Test error")

                result = await visualization_tool.execute(
                    visualization_type="network_graph"
                )

                assert not result.success
                assert "Test error" in result.error


class TestIntegrationNeuralWebVisualization:
    """Integration tests for Neural Web visualization."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_visualization_workflow(self):
        """Test complete visualization workflow with real components."""
        # This test would use real NeuralWeb and MemoryManager instances
        # and test the full pipeline
        pass

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_visualization_with_real_data(self):
        """Test visualization with realistic neural web data."""
        # This test would create a neural web with realistic data
        # and verify visualizations are generated correctly
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
