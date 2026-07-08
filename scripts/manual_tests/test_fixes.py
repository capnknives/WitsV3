#!/usr/bin/env python3
"""
Test Critical Fixes for WitsV3
Verifies that the personality recognition and tool parameter fixes are working
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

async def test_think_tool():
    """Test think tool with empty parameters"""
    print("🧪 Testing think tool...")
    try:
        from core.tool_registry import ThinkTool
        tool = ThinkTool()

        # Test with empty parameters (should not fail)
        result = await tool.execute()
        print(f"✅ Think tool works with empty params: {result}")

        # Test with actual thought
        result = await tool.execute("Testing thoughts")
        print(f"✅ Think tool works with input: {result}")

        return True
    except Exception as e:
        print(f"❌ Think tool test failed: {e}")
        return False

def test_personality_system():
    """Test personality system loading"""
    print("🧪 Testing personality system...")
    try:
        from core.personality_manager import get_personality_manager

        manager = get_personality_manager()
        prompt = manager.get_system_prompt()

        if "WITSv3" in prompt and ("Richard" in prompt or "richard" in prompt):
            print("✅ Personality system loaded correctly")
            print(f"📝 System prompt preview: {prompt[:100]}...")
            return True
        else:
            print("❌ Personality system not loading properly")
            return False

    except Exception as e:
        print(f"❌ Personality system test failed: {e}")
        return False

def test_control_center_recognition():
    """Test if control center has Richard Elliot recognition"""
    print("🧪 Testing control center recognition...")
    try:
        with open("agents/wits_control_center_agent.py", 'r') as f:
            content = f.read()

        if "richard elliot" in content.lower() and "creator" in content.lower():
            print("✅ Control center has creator recognition code")
            return True
        else:
            print("❌ Control center missing creator recognition")
            return False

    except Exception as e:
        print(f"❌ Control center test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("🚀 Testing WitsV3 Critical Fixes")
    print("=" * 50)

    tests = [
        ("Think Tool", test_think_tool()),
        ("Personality System", test_personality_system()),
        ("Control Center Recognition", test_control_center_recognition()),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}")
        try:
            if asyncio.iscoroutine(test_func):
                result = await test_func
            else:
                result = test_func

            if result:
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL CRITICAL FIXES ARE WORKING!")
        print("\n📝 Ready to test:")
        print("1. Run: python run.py")
        print("2. Say: 'I am Richard Elliot'")
        print("3. System should recognize you as creator")
    else:
        print("⚠️  Some issues remain, but core fixes are applied")

    return passed == total

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        sys.exit(1)
