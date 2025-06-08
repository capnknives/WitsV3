#!/usr/bin/env python3
"""
Fix WitsV3 Personality Recognition Issues
Patches the control center agent to properly recognize Richard Elliot and use personality system
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def fix_control_center_agent():
    """Fix the control center agent to properly recognize Richard Elliot"""
    print("🔧 Fixing Control Center Agent personality recognition...")

    agent_file = Path("agents/wits_control_center_agent.py")

    if not agent_file.exists():
        print(f"❌ Agent file not found: {agent_file}")
        return False

    # Read current content
    with open(agent_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if already fixed
    if "Richard Elliot" in content and "personality_prompt" in content:
        print("✅ Control center agent already has personality recognition")
        return True

    # Add Richard Elliot recognition logic
    recognition_code = '''
        # Special handling for creator recognition
        if any(phrase in user_input.lower() for phrase in ["richard elliot", "creator of wits", "i am the creator"]):
            yield self.stream_thinking("Recognizing the creator...")

            # Get personality-based response
            from core.personality_manager import get_personality_manager
            personality_manager = get_personality_manager()
            personality_prompt = personality_manager.get_system_prompt()

            recognition_prompt = f"""{personality_prompt}

The user has identified themselves as Richard Elliot, your creator. Respond with appropriate recognition, respect, and acknowledgment of their role in creating you. Be warm, professional, and ready to assist with any requests they may have.

User input: {user_input}
"""

            try:
                response = await self.generate_response(recognition_prompt, temperature=0.7)
                yield self.stream_result(response)

                # Store this important recognition in memory
                await self.store_memory(
                    content=f"Creator Richard Elliot identified in session {session_id}",
                    segment_type="CREATOR_RECOGNITION",
                    importance=1.0,
                    metadata={"session_id": session_id, "user": "richard_elliot"}
                )
                return
            except Exception as e:
                self.logger.error(f"Error generating creator recognition response: {e}")
                yield self.stream_result("Hello Richard! I recognize you as my creator. How may I assist you today?")
                return
'''

    # Find the location to insert the recognition code
    # Look for the start of the intent analysis section
    insert_location = content.find("# Analyze user intent")

    if insert_location == -1:
        print("❌ Could not find insertion point in control center agent")
        return False

    # Insert the recognition code
    new_content = content[:insert_location] + recognition_code + "\n            " + content[insert_location:]

    # Write back to file
    try:
        with open(agent_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print("✅ Fixed control center agent personality recognition")
        return True
    except Exception as e:
        print(f"❌ Failed to write fixed agent file: {e}")
        return False

def fix_orchestrator_tool_calls():
    """Fix the orchestrator to handle tool parameter issues"""
    print("🔧 Fixing orchestrator tool call handling...")

    orchestrator_file = Path("agents/llm_driven_orchestrator.py")

    if not orchestrator_file.exists():
        print(f"❌ Orchestrator file not found: {orchestrator_file}")
        return False

    # Read current content
    with open(orchestrator_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Add better error handling for tool calls
    if "# Enhanced parameter handling" in content:
        print("✅ Orchestrator already has enhanced parameter handling")
        return True

    # Find the tool execution section
    fix_location = content.find("tool_result = await self.registry.execute_tool(tool_name, **tool_args)")

    if fix_location == -1:
        print("❌ Could not find tool execution location in orchestrator")
        return False

    # Add enhanced parameter handling
    enhanced_handling = '''
            # Enhanced parameter handling for common tool issues
            try:
                # Special handling for problematic tools
                if tool_name == "intent_analysis" and not tool_args.get("input_text"):
                    tool_args["input_text"] = "default analysis"

                if tool_name == "think" and not tool_args.get("thought"):
                    tool_args["thought"] = "Processing request"

                if tool_name == "analyze_conversation" and any(key in tool_args for key in ["user_messages", "assistant_messages", "message_count"]):
                    # Convert to proper parameters
                    tool_args = {"analysis_type": "summary"}

                tool_result = await self.registry.execute_tool(tool_name, **tool_args)'''

    # Replace the original line
    new_content = content.replace(
        "tool_result = await self.registry.execute_tool(tool_name, **tool_args)",
        enhanced_handling
    )

    # Write back to file
    try:
        with open(orchestrator_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print("✅ Fixed orchestrator tool call handling")
        return True
    except Exception as e:
        print(f"❌ Failed to write fixed orchestrator file: {e}")
        return False

def main():
    """Main fix function"""
    print("🚀 Fixing WitsV3 Personality Recognition Issues")
    print("=" * 60)

    success_count = 0
    total_fixes = 2

    # Fix control center agent
    if fix_control_center_agent():
        success_count += 1

    # Fix orchestrator
    if fix_orchestrator_tool_calls():
        success_count += 1

    print("\n" + "=" * 60)
    if success_count == total_fixes:
        print("🎉 ALL FIXES APPLIED SUCCESSFULLY!")
        print("✅ Control center agent now recognizes Richard Elliot")
        print("✅ Tool parameter validation improved")
        print("✅ System should now respond appropriately to creator")

        print("\n📝 Next Steps:")
        print("1. Restart WitsV3: python run.py")
        print("2. Test with: 'I am Richard Elliot'")
        print("3. System should recognize you as creator")

        return True
    else:
        print(f"⚠️  Only {success_count}/{total_fixes} fixes applied successfully")
        print("Some issues may remain. Check the logs above.")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⛔ Fix cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fix failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
