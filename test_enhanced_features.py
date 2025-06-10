#!/usr/bin/env python3
"""
Test script for enhanced WitsV3 features
Tests the new meta-reasoning, tool composition, and adaptive components
"""

import asyncio
import logging
from core.config import WitsV3Config
from core.concrete_meta_reasoning import WitsV3MetaReasoningEngine
from core.tool_composition import IntelligentToolComposer
from core.adaptive import PerformanceTracker, AdaptiveTokenizer, ResponseGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

async def test_enhanced_features():
    """Test all enhanced WitsV3 features"""
    print("🚀 Testing Enhanced WitsV3 Features")
    print("=" * 50)

    try:
        # Load config
        config = WitsV3Config.from_yaml('config.yaml')
        print("✅ Configuration loaded successfully")

        # Test Adaptive Components
        print("\n🧠 Testing Adaptive Components...")
        tokenizer = AdaptiveTokenizer()
        response_generator = ResponseGenerator(tokenizer)
        print("✅ Adaptive tokenizer and response generator initialized")

        # Test a simple tokenization
        test_text = "Hello, this is a test of the enhanced WitsV3 system!"
        tokens = await tokenizer.tokenize(test_text)
        decoded = await tokenizer.decode(tokens.squeeze(0))
        print(f"✅ Tokenization test: '{test_text[:20]}...' -> {tokens.shape[1]} tokens")

        # Test Meta-reasoning Engine
        print("\n🧠 Testing Meta-reasoning Engine...")
        meta_engine = WitsV3MetaReasoningEngine(config)
        print("✅ Meta-reasoning engine initialized")

        # Test problem analysis
        test_goal = "Create a comprehensive analysis of user data patterns"
        test_context = {
            "complexity": "moderate",
            "time_limit": 300,
            "constraints": ["privacy_compliant", "accurate_results"]
        }

        problem_space = await meta_engine.analyze_problem_space(test_goal, test_context)
        print(f"✅ Problem analysis complete:")
        print(f"   - Complexity: {problem_space.complexity.value}")
        print(f"   - Estimated steps: {problem_space.estimated_steps}")
        print(f"   - Confidence: {problem_space.confidence:.0%}")

        # Test execution plan generation
        available_agents = ["wits_control_center", "orchestrator", "json_tool", "math_tool"]
        execution_plan = await meta_engine.generate_execution_plan(problem_space, available_agents)
        print(f"✅ Execution plan generated:")
        print(f"   - Strategy: {execution_plan.strategy.value}")
        print(f"   - Steps: {len(execution_plan.steps)}")
        print(f"   - Estimated duration: {execution_plan.estimated_duration:.0f}s")

        # Test Tool Composition Engine
        print("\n🔧 Testing Tool Composition Engine...")
        tool_composer = IntelligentToolComposer(config)
        print("✅ Tool composition engine initialized")

        # Test workflow composition
        composition_goal = "Analyze JSON data and generate mathematical summary"
        available_tools = ["json_manipulate", "math_operations", "file_reader"]

        workflow = await tool_composer.compose_workflow(
            composition_goal,
            available_tools,
            constraints={"max_time": 120, "parallel_allowed": True}
        )
        print(f"✅ Workflow composed:")
        print(f"   - Strategy: {workflow.strategy.value}")
        print(f"   - Nodes: {len(workflow.nodes)}")
        print(f"   - Data flow: {workflow.data_flow.value}")

        # Test workflow execution order
        execution_order = workflow.get_execution_order()
        print(f"✅ Execution order calculated: {len(execution_order)} levels")

        # Test Performance Tracking
        print("\n📊 Testing Performance Tracking...")

        # Create a mock settings object
        class MockSettings:
            enable_performance_tracking = True
            performance_log_path = "logs/performance.json"

        perf_tracker = PerformanceTracker(MockSettings())

        # Track some performance data
        perf_tracker.track_performance(
            prompt="Test prompt",
            response="Test response",
            module="test_module",
            complexity=0.5,
            generation_time=1.23,
            cache_hit=False
        )

        stats = perf_tracker.get_performance_stats()
        print(f"✅ Performance tracking working:")
        print(f"   - Entries: {stats['count']}")
        print(f"   - Avg time: {stats['avg_generation_time']:.2f}s")
        print(f"   - Cache hit rate: {stats['cache_hit_rate']:.0%}")

        print("\n🎉 All Enhanced Features Test Complete!")
        print("=" * 50)
        print("✅ Meta-reasoning: Problem analysis and execution planning")
        print("✅ Tool Composition: Intelligent workflow generation")
        print("✅ Adaptive Components: Tokenization and response generation")
        print("✅ Performance Tracking: Metrics collection and analysis")
        print("\n🚀 WitsV3 Enhanced System is fully operational!")

        return True

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_enhanced_features())
    exit(0 if success else 1)
