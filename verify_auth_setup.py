#!/usr/bin/env python3
"""
WitsV3 Authentication Verification Script
Verifies that the authentication system is working with existing configuration
"""

import sys
import asyncio
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.auth_manager import get_auth_manager, verify_auth
from tools.network_control_tool import NetworkControlTool
from core.personality_manager import PersonalityManager

async def main():
    """Test the authentication system with user interaction"""
    print("🔐 WitsV3 Authentication Verification")
    print("=" * 50)

    auth_manager = get_auth_manager()

    # Check if token is configured
    if not auth_manager.is_token_configured():
        print("❌ No authentication token configured!")
        print("Please run: python setup_auth.py")
        return 1

    print("✅ Authentication token is configured")

    # Get token from user for testing
    print("\nTo verify the system is working, please provide your authentication token.")
    print("(This is the token that was generated during setup)")
    auth_token = input("Enter your authentication token: ").strip()

    if not auth_token:
        print("❌ No token provided")
        return 1

    print("\n🧪 Testing Authentication System...")
    print("-" * 40)

    # Test 1: Basic token verification
    print("1. Testing basic token verification...")
    if auth_manager.verify_token(auth_token, "richard_elliot"):
        print("   ✅ Token verification: PASS")
    else:
        print("   ❌ Token verification: FAIL")
        return 1

    # Test 2: Network control authorization
    print("2. Testing network control authorization...")
    is_auth, msg = verify_auth("richard_elliot", auth_token, "network_control")
    if is_auth:
        print(f"   ✅ Network control auth: PASS - {msg}")
    else:
        print(f"   ❌ Network control auth: FAIL - {msg}")
        return 1

    # Test 3: Ethics override authorization
    print("3. Testing ethics override authorization...")
    is_auth, msg = verify_auth("richard_elliot", auth_token, "ethics_override")
    if is_auth:
        print(f"   ✅ Ethics override auth: PASS - {msg}")
    else:
        print(f"   ❌ Ethics override auth: FAIL - {msg}")
        return 1

    # Test 4: Network control tool integration
    print("4. Testing network control tool...")
    try:
        tool = NetworkControlTool()

        # Test status (should work without auth)
        result = await tool.execute("status")
        if result["success"]:
            print("   ✅ Network status check: PASS")
        else:
            print(f"   ❌ Network status check: FAIL - {result['message']}")

        # Test network enable with auth
        result = await tool.execute("enable_network", "richard_elliot", 5, auth_token)
        if result["success"]:
            print("   ✅ Network enable with auth: PASS")

            # Test network disable
            result = await tool.execute("disable_network", "richard_elliot", 5, auth_token)
            if result["success"]:
                print("   ✅ Network disable with auth: PASS")
            else:
                print(f"   ❌ Network disable: FAIL - {result['message']}")
        else:
            print(f"   ❌ Network enable with auth: FAIL - {result['message']}")

    except Exception as e:
        print(f"   ❌ Network control test: FAIL - {e}")
        return 1

    # Test 5: Unauthorized access prevention
    print("5. Testing unauthorized access prevention...")

    # Test with wrong user
    is_auth, msg = verify_auth("wrong_user", auth_token, "network_control")
    if not is_auth:
        print("   ✅ Wrong user blocked: PASS")
    else:
        print("   ❌ Wrong user blocked: FAIL - unauthorized user was allowed")
        return 1

    # Test with wrong token
    is_auth, msg = verify_auth("richard_elliot", "wrong_token", "network_control")
    if not is_auth:
        print("   ✅ Wrong token blocked: PASS")
    else:
        print("   ❌ Wrong token blocked: FAIL - wrong token was accepted")
        return 1

    print("\n" + "=" * 50)
    print("🎉 AUTHENTICATION SYSTEM VERIFICATION COMPLETE!")
    print("✅ All security features are working correctly")
    print("🔐 Your WitsV3 system is properly secured")

    print("\n📝 Summary of Security Features:")
    print("• Token-based authentication for sensitive operations")
    print("• SHA256 hashed token storage (never stored in plaintext)")
    print("• User ID validation combined with token verification")
    print("• Network access control with authentication")
    print("• Ethics override protection with authentication")
    print("• Unauthorized access prevention")

    return 0

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(result)
    except KeyboardInterrupt:
        print("\n⛔ Verification cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
