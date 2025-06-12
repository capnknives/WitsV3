"""
Neural Web Visualization Tools for WitsV3

Provides visualization capabilities for the Neural Web knowledge networks
including static graphs and interactive HTML visualizations.
"""

import logging
import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

from core.base_tool import BaseTool, ToolResult
from core.config import WitsV3Config
from core.neural_web_core import NeuralWeb
from core.memory_manager import MemoryManager

logger = logging.getLogger(__name__)


class NeuralWebVisualizer:
    """Visualization tools for Neural Web knowledge networks."""

    def __init__(self, neural_web: NeuralWeb, config: WitsV3Config):
        self.neural_web = neural_web
        self.config = config
        self.logger = logging.getLogger(__name__)

    async def generate_network_graph(self,
                                   format: str = "png",
                                   include_weights: bool = True,
                                   filter_threshold: float = 0.2,
                                   max_nodes: int = 100,
                                   include_domains: bool = True) -> str:
        """
        Generate a visualization of the neural web network.

        Args:
            format: Output format (png, svg, pdf)
            include_weights: Whether to include connection weights
            filter_threshold: Minimum connection strength to include
            max_nodes: Maximum number of nodes to include
            include_domains: Whether to color-code by domain

        Returns:
            Path to the generated visualization file
        """
        try:
            # Create a filtered graph for visualization
            viz_graph = nx.DiGraph()

            # Filter concepts by activation level and max_nodes
            concept_items = list(self.neural_web.concepts.items())
            if len(concept_items) > max_nodes:
                # Sort by activation level and take top max_nodes
                concept_items.sort(key=lambda x: x[1].activation_level, reverse=True)
                concept_items = concept_items[:max_nodes]

            # Add nodes
            node_colors = []
            node_labels = {}
            domain_color_map = {}
            color_palette = plt.cm.Set3(np.linspace(0, 1, 12))  # 12 distinct colors

            for i, (concept_id, concept) in enumerate(concept_items):
                viz_graph.add_node(concept_id)

                # Create label (truncate if too long)
                label = concept.content[:20] + "..." if len(concept.content) > 20 else concept.content
                node_labels[concept_id] = label

                # Color by domain if enabled
                if include_domains and "domain" in concept.metadata:
                    domain = concept.metadata["domain"]
                    if domain not in domain_color_map:
                        color_idx = len(domain_color_map) % len(color_palette)
                        domain_color_map[domain] = color_palette[color_idx]
                    node_colors.append(domain_color_map[domain])
                else:
                    # Default color based on activation level
                    activation = concept.activation_level
                    node_colors.append(plt.cm.Reds(0.3 + 0.7 * activation))

            # Add edges with filtering
            edge_weights = []
            for (source_id, target_id), connection in self.neural_web.connections.items():
                if (source_id in viz_graph.nodes and
                    target_id in viz_graph.nodes and
                    connection.strength >= filter_threshold):

                    viz_graph.add_edge(source_id, target_id)
                    edge_weights.append(connection.strength * 5)  # Scale for visibility

            # Create the plot
            plt.figure(figsize=(16, 12))

            # Use spring layout for better organization
            try:
                pos = nx.spring_layout(viz_graph, k=3, iterations=50)
            except:
                # Fallback to circular layout if spring fails
                pos = nx.circular_layout(viz_graph)

            # Draw the graph
            nx.draw_networkx_nodes(
                viz_graph, pos,
                node_color=node_colors,
                node_size=500,
                alpha=0.8
            )

            # Draw edges with varying thickness if weights included
            if include_weights and edge_weights:
                nx.draw_networkx_edges(
                    viz_graph, pos,
                    width=edge_weights,
                    alpha=0.6,
                    edge_color='gray',
                    arrows=True,
                    arrowsize=20
                )
            else:
                nx.draw_networkx_edges(
                    viz_graph, pos,
                    alpha=0.6,
                    edge_color='gray',
                    arrows=True,
                    arrowsize=20
                )

            # Add labels
            nx.draw_networkx_labels(
                viz_graph, pos,
                labels=node_labels,
                font_size=8,
                font_weight='bold'
            )

            # Add title
            plt.title(f"Neural Web Knowledge Network\n{len(viz_graph.nodes)} concepts, {len(viz_graph.edges)} connections",
                     fontsize=16, fontweight='bold')

            # Add legend for domains if enabled
            if include_domains and domain_color_map:
                patches = [mpatches.Patch(color=color, label=domain)
                          for domain, color in domain_color_map.items()]
                plt.legend(handles=patches, loc='upper right', bbox_to_anchor=(1.0, 1.0))

            # Remove axes
            plt.axis('off')

            # Save visualization
            output_dir = Path("data/visualizations")
            output_dir.mkdir(exist_ok=True, parents=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"neural_web_{timestamp}.{format}"

            plt.savefig(output_path, format=format, dpi=300, bbox_inches='tight')
            plt.close()

            self.logger.info(f"Generated neural web visualization: {output_path}")
            return str(output_path)

        except Exception as e:
            self.logger.error(f"Error generating network graph: {e}")
            raise

    async def generate_interactive_visualization(self,
                                               max_nodes: int = 100,
                                               include_domains: bool = True,
                                               include_weights: bool = True) -> str:
        """
        Generate an interactive HTML visualization of the neural web.

        Args:
            max_nodes: Maximum number of nodes to include
            include_domains: Whether to include domain information
            include_weights: Whether to include connection weights

        Returns:
            Path to the generated HTML file
        """
        try:
            # Prepare data for D3.js visualization
            nodes = []
            links = []

            # Filter and prepare nodes
            concept_items = list(self.neural_web.concepts.items())
            if len(concept_items) > max_nodes:
                concept_items.sort(key=lambda x: x[1].activation_level, reverse=True)
                concept_items = concept_items[:max_nodes]

            concept_ids = {concept_id for concept_id, _ in concept_items}

            # Create nodes
            for concept_id, concept in concept_items:
                node_data = {
                    "id": concept_id,
                    "label": concept.content[:30] + "..." if len(concept.content) > 30 else concept.content,
                    "activation": concept.activation_level,
                    "type": concept.concept_type,
                    "content": concept.content
                }

                if include_domains and "domain" in concept.metadata:
                    node_data["domain"] = concept.metadata["domain"]

                nodes.append(node_data)

            # Create links
            for (source_id, target_id), connection in self.neural_web.connections.items():
                if source_id in concept_ids and target_id in concept_ids:
                    link_data = {
                        "source": source_id,
                        "target": target_id,
                        "relationship": connection.relationship_type,
                        "strength": connection.strength,
                        "confidence": connection.confidence
                    }

                    if include_weights:
                        link_data["weight"] = connection.strength

                    links.append(link_data)

            # Create HTML content
            html_content = self._create_interactive_html(nodes, links, include_domains)

            # Save HTML file
            output_dir = Path("data/visualizations")
            output_dir.mkdir(exist_ok=True, parents=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"neural_web_interactive_{timestamp}.html"

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            self.logger.info(f"Generated interactive neural web visualization: {output_path}")
            return str(output_path)

        except Exception as e:
            self.logger.error(f"Error generating interactive visualization: {e}")
            raise

    def _create_interactive_html(self, nodes: List[Dict], links: List[Dict], include_domains: bool) -> str:
        """Create HTML content for interactive visualization."""

        nodes_json = json.dumps(nodes, indent=2)
        links_json = json.dumps(links, indent=2)

        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Neural Web Interactive Visualization</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            font-family: Arial, sans-serif;
            background-color: #1a1a1a;
            color: white;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}

        .title {{
            text-align: center;
            margin-bottom: 20px;
            font-size: 24px;
            font-weight: bold;
        }}

        .controls {{
            margin-bottom: 20px;
            text-align: center;
        }}

        .controls button {{
            margin: 5px;
            padding: 8px 16px;
            background-color: #333;
            color: white;
            border: 1px solid #555;
            border-radius: 4px;
            cursor: pointer;
        }}

        .controls button:hover {{
            background-color: #555;
        }}

        .visualization {{
            border: 1px solid #333;
            border-radius: 8px;
            background-color: #2a2a2a;
        }}

        .node {{
            stroke: #fff;
            stroke-width: 2px;
            cursor: pointer;
        }}

        .link {{
            stroke: #999;
            stroke-opacity: 0.6;
            marker-end: url(#end);
        }}

        .node-label {{
            fill: white;
            font-size: 12px;
            text-anchor: middle;
            pointer-events: none;
        }}

        .tooltip {{
            position: absolute;
            padding: 10px;
            background-color: rgba(0, 0, 0, 0.9);
            border: 1px solid #555;
            border-radius: 4px;
            color: white;
            font-size: 12px;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.3s;
        }}

        .stats {{
            margin-top: 20px;
            padding: 15px;
            background-color: #2a2a2a;
            border-radius: 8px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="title">Neural Web Interactive Visualization</div>

        <div class="controls">
            <button onclick="restartSimulation()">Restart Layout</button>
            <button onclick="toggleDomains()">Toggle Domains</button>
            <button onclick="toggleWeights()">Toggle Weights</button>
            <button onclick="zoomFit()">Zoom to Fit</button>
        </div>

        <div class="visualization">
            <svg id="visualization" width="1160" height="600"></svg>
        </div>

        <div class="stats">
            <div><strong>Network Statistics:</strong></div>
            <div id="stats-content"></div>
        </div>
    </div>

    <div class="tooltip" id="tooltip"></div>

    <script>
        // Data
        const nodes = {nodes_json};
        const links = {links_json};

        // Configuration
        let showDomains = {str(include_domains).lower()};
        let showWeights = true;

        // SVG setup
        const svg = d3.select("#visualization");
        const width = +svg.attr("width");
        const height = +svg.attr("height");

        // Create arrow marker
        svg.append("defs").append("marker")
            .attr("id", "end")
            .attr("viewBox", "0 -5 10 10")
            .attr("refX", 20)
            .attr("refY", 0)
            .attr("markerWidth", 6)
            .attr("markerHeight", 6)
            .attr("orient", "auto")
            .append("path")
            .attr("d", "M0,-5L10,0L0,5")
            .attr("fill", "#999");

        // Create simulation
        const simulation = d3.forceSimulation(nodes)
            .force("link", d3.forceLink(links).id(d => d.id).distance(100))
            .force("charge", d3.forceManyBody().strength(-300))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collision", d3.forceCollide().radius(25));

        // Create zoom behavior
        const zoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on("zoom", zoomed);

        svg.call(zoom);

        // Create container for zoom/pan
        const container = svg.append("g");

        function zoomed(event) {{
            container.attr("transform", event.transform);
        }}

        // Color scale for domains
        const domainColors = d3.scaleOrdinal(d3.schemeCategory10);

        // Create links
        const link = container.append("g")
            .selectAll("line")
            .data(links)
            .enter().append("line")
            .attr("class", "link")
            .style("stroke-width", d => showWeights ? Math.sqrt(d.strength * 10) : 2);

        // Create nodes
        const node = container.append("g")
            .selectAll("circle")
            .data(nodes)
            .enter().append("circle")
            .attr("class", "node")
            .attr("r", 15)
            .style("fill", d => {{
                if (showDomains && d.domain) {{
                    return domainColors(d.domain);
                }} else {{
                    return d3.interpolateReds(0.3 + 0.7 * d.activation);
                }}
            }})
            .on("mouseover", handleMouseOver)
            .on("mouseout", handleMouseOut)
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended));

        // Create labels
        const label = container.append("g")
            .selectAll("text")
            .data(nodes)
            .enter().append("text")
            .attr("class", "node-label")
            .text(d => d.label);

        // Update simulation
        simulation.on("tick", () => {{
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);

            node
                .attr("cx", d => d.x)
                .attr("cy", d => d.y);

            label
                .attr("x", d => d.x)
                .attr("y", d => d.y + 5);
        }});

        // Event handlers
        function handleMouseOver(event, d) {{
            const tooltip = d3.select("#tooltip");
            tooltip.style("opacity", 1)
                .html(`
                    <strong>${{d.label}}</strong><br/>
                    Type: ${{d.type}}<br/>
                    Activation: ${{d.activation.toFixed(3)}}<br/>
                    ${{d.domain ? `Domain: ${{d.domain}}<br/>` : ''}}
                    <hr style="margin: 5px 0;">
                    ${{d.content}}
                `)
                .style("left", (event.pageX + 10) + "px")
                .style("top", (event.pageY - 10) + "px");
        }}

        function handleMouseOut() {{
            d3.select("#tooltip").style("opacity", 0);
        }}

        function dragstarted(event, d) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }}

        function dragged(event, d) {{
            d.fx = event.x;
            d.fy = event.y;
        }}

        function dragended(event, d) {{
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }}

        // Control functions
        function restartSimulation() {{
            simulation.alpha(1).restart();
        }}

        function toggleDomains() {{
            showDomains = !showDomains;
            node.style("fill", d => {{
                if (showDomains && d.domain) {{
                    return domainColors(d.domain);
                }} else {{
                    return d3.interpolateReds(0.3 + 0.7 * d.activation);
                }}
            }});
        }}

        function toggleWeights() {{
            showWeights = !showWeights;
            link.style("stroke-width", d => showWeights ? Math.sqrt(d.strength * 10) : 2);
        }}

        function zoomFit() {{
            const bounds = container.node().getBBox();
            const fullWidth = width;
            const fullHeight = height;
            const midX = bounds.x + bounds.width / 2;
            const midY = bounds.y + bounds.height / 2;

            const scale = 0.8 / Math.max(bounds.width / fullWidth, bounds.height / fullHeight);
            const translate = [fullWidth / 2 - scale * midX, fullHeight / 2 - scale * midY];

            svg.transition()
                .duration(750)
                .call(zoom.transform, d3.zoomIdentity.translate(translate[0], translate[1]).scale(scale));
        }}

        // Update statistics
        function updateStats() {{
            const domains = new Set(nodes.filter(n => n.domain).map(n => n.domain));
            const avgActivation = nodes.reduce((sum, n) => sum + n.activation, 0) / nodes.length;

            document.getElementById("stats-content").innerHTML = `
                <div>Nodes: ${{nodes.length}}</div>
                <div>Links: ${{links.length}}</div>
                <div>Domains: ${{domains.size}}</div>
                <div>Average Activation: ${{avgActivation.toFixed(3)}}</div>
                <div>Network Density: ${{(2 * links.length / (nodes.length * (nodes.length - 1))).toFixed(3)}}</div>
            `;
        }}

        updateStats();
    </script>
</body>
</html>
        """

        return html_template


class NeuralWebVisualizationTool(BaseTool):
    """Tool for generating Neural Web visualizations."""

    def __init__(self, config: WitsV3Config):
        super().__init__(config)
        self.name = "neural_web_visualize"
        self.description = "Generate visualizations of the Neural Web knowledge network"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "format": {
                            "type": "string",
                            "enum": ["png", "svg", "pdf", "html"],
                            "description": "Output format for the visualization"
                        },
                        "include_weights": {
                            "type": "boolean",
                            "description": "Whether to include connection weights"
                        },
                        "filter_threshold": {
                            "type": "number",
                            "description": "Minimum connection strength to include"
                        },
                        "max_nodes": {
                            "type": "integer",
                            "description": "Maximum number of nodes to include"
                        },
                        "include_domains": {
                            "type": "boolean",
                            "description": "Whether to include domain information"
                        }
                    },
                    "required": ["format"]
                }
            }
        }

    async def execute(self, **kwargs) -> ToolResult:
        try:
            # Get Neural Web instance
            neural_web = await self._get_neural_web()

            if not neural_web:
                return ToolResult(
                    success=False,
                    error="Neural Web not available or empty"
                )

            # Create visualizer
            visualizer = NeuralWebVisualizer(neural_web, self.config)

            # Generate visualization
            format = kwargs.get("format", "png")

            if format == "html":
                max_nodes = kwargs.get("max_nodes", 100)
                include_domains = kwargs.get("include_domains", True)
                include_weights = kwargs.get("include_weights", True)

                output_path = await visualizer.generate_interactive_visualization(
                    max_nodes=max_nodes,
                    include_domains=include_domains,
                    include_weights=include_weights
                )
            else:
                include_weights = kwargs.get("include_weights", True)
                filter_threshold = kwargs.get("filter_threshold", 0.2)
                max_nodes = kwargs.get("max_nodes", 100)
                include_domains = kwargs.get("include_domains", True)

                output_path = await visualizer.generate_network_graph(
                    format=format,
                    include_weights=include_weights,
                    filter_threshold=filter_threshold,
                    max_nodes=max_nodes,
                    include_domains=include_domains
                )

            return ToolResult(
                success=True,
                result=f"Neural Web visualization created at: {output_path}",
                metadata={
                    "path": output_path,
                    "format": format,
                    "node_count": len(neural_web.concepts),
                    "connection_count": len(neural_web.connections)
                }
            )

        except Exception as e:
            logger.error(f"Error creating neural web visualization: {e}")
            return ToolResult(
                success=False,
                error=f"Error creating visualization: {str(e)}"
            )

    async def _get_neural_web(self) -> Optional[NeuralWeb]:
        """Get the Neural Web instance from the system."""
        try:
            # For now, create a simple test neural web
            # In production, this would be retrieved from the memory manager or agent
            neural_web = NeuralWeb()

            # Add some test data if the neural web is empty
            if not neural_web.concepts:
                await self._add_test_data(neural_web)

            return neural_web

        except Exception as e:
            logger.error(f"Error getting neural web: {e}")
            return None

    async def _add_test_data(self, neural_web: NeuralWeb):
        """Add some test data to demonstrate visualization."""
        try:
            # Add test concepts
            test_concepts = [
                ("python", "Python programming language", "technology", {"domain": "computer_science"}),
                ("ai", "Artificial Intelligence", "technology", {"domain": "computer_science"}),
                ("machine_learning", "Machine Learning algorithms", "technology", {"domain": "computer_science"}),
                ("creativity", "Human creativity and innovation", "concept", {"domain": "psychology"}),
                ("art", "Artistic expression and creation", "concept", {"domain": "arts"}),
                ("music", "Musical composition and performance", "concept", {"domain": "arts"}),
                ("physics", "Laws of physics and matter", "science", {"domain": "physics"}),
                ("gravity", "Gravitational force", "science", {"domain": "physics"}),
                ("democracy", "Democratic governance system", "concept", {"domain": "politics"}),
                ("freedom", "Individual freedom and rights", "concept", {"domain": "politics"}),
            ]

            for concept_id, content, concept_type, metadata in test_concepts:
                await neural_web.add_concept(concept_id, content, concept_type, metadata)
                neural_web.concepts[concept_id].activation_level = np.random.uniform(0.2, 0.9)

            # Add test connections
            test_connections = [
                ("python", "ai", "enables", 0.8),
                ("ai", "machine_learning", "includes", 0.9),
                ("python", "machine_learning", "implements", 0.7),
                ("creativity", "art", "enables", 0.8),
                ("art", "music", "related_to", 0.6),
                ("creativity", "ai", "inspires", 0.5),
                ("physics", "gravity", "includes", 0.9),
                ("democracy", "freedom", "enables", 0.8),
                ("creativity", "freedom", "requires", 0.6),
                ("ai", "creativity", "mimics", 0.4),
            ]

            for source, target, relationship, strength in test_connections:
                await neural_web.connect_concepts(source, target, relationship, strength)

            logger.info("Added test data to neural web for visualization")

        except Exception as e:
            logger.error(f"Error adding test data: {e}")
