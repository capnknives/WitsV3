#!/usr/bin/env python3
"""
Comprehensive test suite for WitsV3 Authentication System
Tests secure token authentication for network control and ethics override
"""

import asyncio
import tempfile
import os
import yaml
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.auth_manager import AuthManager, generate_new_token, verify_auth
from tools.network_control_tool import NetworkControlTool
from core.personality_manager import PersonalityManager

class TestAuthenticationSystem:
    """Test suite for authentication system"""

    def __init__(self):
        self.temp_config = None
        self.original_config = None
        self.test_token = None
        self.test_token_hash = None

    async def setup_test_environment(self):
        """Set up isolated test environment"""
        print("🔧 Setting up test environment...")

        # Generate test token
        auth_manager = AuthManager()
        self.test_token = generate_new_token()
        self.test_token_hash = auth_manager.hash_token(self.test_token)

        # Create temporary config for testing
        self.temp_config = {
            'security': {
                'python_execution_network_access': False,
                'python_execution_subprocess_access': False,
                'authorized_network_override_user': 'richard_elliot',
                'ethics_system_enabled': True,
                'ethics_override_authorized_user': 'richard_elliot',
                'auth_token_hash': self.test_token_hash,
                'require_auth_for_network_control': True,
                'require_auth_for_ethics_override': True
            }
        }

        # Backup original config
        if os.path.exists('config.yaml'):
            with open('config.yaml', 'r') as f:
                self.original_config = f.read()

        # Write test config
        with open('config.yaml', 'w') as f:
            yaml.safe_dump(self.temp_config, f)

        print(f"✅ Test token generated: {self.test_token[:16]}...")
        print(f"✅ Test environment configured")

    async def cleanup_test_environment(self):
        """Clean up test environment"""
        print("🧹 Cleaning up test environment...")

        # Restore original config
        if self.original_config:
            with open('config.yaml', 'w') as f:
                f.write(self.original_config)

        print("✅ Test environment cleaned up")

    async def test_token_generation_and_hashing(self):
        """Test token generation and hashing"""
        print("\n🔐 Testing token generation and hashing...")

        auth_manager = AuthManager()

        # Test token generation
        token1 = auth_manager.generate_token()
        token2 = auth_manager.generate_token()

        assert len(token1) == 64, f"Expected token length 64, got {len(token1)}"
        assert len(token2) == 64, f"Expected token length 64, got {len(token2)}"
        assert token1 != token2, "Tokens should be unique"

        # Test token hashing
        hash1 = auth_manager.hash_token(token1)
        hash2 = auth_manager.hash_token(token1)  # Same token
        hash3 = auth_manager.hash_token(token2)  # Different token

        assert len(hash1) == 64, f"Expected hash length 64, got {len(hash1)}"
        assert hash1 == hash2, "Same token should produce same hash"
        assert hash1 != hash3, "Different tokens should produce different hashes"

        print("✅ Token generation and hashing tests passed")

    async def test_token_verification(self):
        """Test token verification"""
        print("\n🔍 Testing token verification...")

        auth_manager = AuthManager()

        # Test with correct token
        assert auth_manager.verify_token(self.test_token, "richard_elliot"), "Valid token should verify"

        # Test with incorrect token
        wrong_token = generate_new_token()
        assert not auth_manager.verify_token(wrong_token, "richard_elliot"), "Invalid token should not verify"

        # Test with empty token
        assert not auth_manager.verify_token("", "richard_elliot"), "Empty token should not verify"

        print("✅ Token verification tests passed")

    async def test_user_and_token_verification(self):
        """Test combined user ID and token verification"""
        print("\n👤 Testing user and token verification...")

        # Test network control authorization
        is_auth, msg = verify_auth("richard_elliot", self.test_token, "network_control")
        assert is_auth, f"Authorized user with valid token should pass: {msg}"

        # Test wrong user
        is_auth, msg = verify_auth("wrong_user", self.test_token, "network_control")
        assert not is_auth, f"Wrong user should fail: {msg}"

        # Test wrong token
        is_auth, msg = verify_auth("richard_elliot", "wrong_token", "network_control")
        assert not is_auth, f"Wrong token should fail: {msg}"

        # Test ethics override authorization
        is_auth, msg = verify_auth("richard_elliot", self.test_token, "ethics_override")
        assert is_auth, f"Authorized user with valid token should pass ethics: {msg}"

        print("✅ User and token verification tests passed")

    async def test_network_control_with_auth(self):
        """Test network control tool with authentication"""
        print("\n🌐 Testing network control with authentication...")

        tool = NetworkControlTool()

        # Test status without auth (should work)
        result = await tool.execute("status")
        assert result["success"], f"Status check should work without auth: {result['message']}"

        # Test enable without auth (should fail)
        result = await tool.execute("enable_network", "richard_elliot")
        assert not result["success"], f"Enable without auth should fail: {result['message']}"

        # Test enable with wrong token (should fail)
        result = await tool.execute("enable_network", "richard_elliot", 60, "wrong_token")
        assert not result["success"], f"Enable with wrong token should fail: {result['message']}"

        # Test enable with correct auth (should work)
        result = await tool.execute("enable_network", "richard_elliot", 60, self.test_token)
        assert result["success"], f"Enable with correct auth should work: {result['message']}"

        # Test disable with correct auth (should work)
        result = await tool.execute("disable_network", "richard_elliot", 60, self.test_token)
        assert result["success"], f"Disable with correct auth should work: {result['message']}"

        print("✅ Network control authentication tests passed")

    async def test_ethics_override_with_auth(self):
        """Test ethics override with authentication"""
        print("\n⚖️  Testing ethics override with authentication...")

        # Create temporary ethics config
        ethics_config = {
            'ethics_overlay': {
                'name': 'Test Ethics Framework',
                'testing_framework': {
                    'authorized_override_user': 'richard_elliot'
                }
            }
        }

        with open('config/ethics_overlay.yaml', 'w') as f:
            yaml.safe_dump(ethics_config, f)

        personality_manager = PersonalityManager()

        # Test override without auth (should fail)
        result = personality_manager.enable_ethics_override("richard_elliot", "testing")
        assert not result, "Ethics override without auth should fail"

        # Test override with wrong token (should fail)
        result = personality_manager.enable_ethics_override("richard_elliot", "testing", 60, "wrong_token")
        assert not result, "Ethics override with wrong token should fail"

        # Test override with correct auth (should work)
        result = personality_manager.enable_ethics_override("richard_elliot", "testing", 60, self.test_token)
        assert result, "Ethics override with correct auth should work"

        print("✅ Ethics override authentication tests passed")

    async def test_security_configuration(self):
        """Test security configuration options"""
        print("\n⚙️  Testing security configuration...")

        # Test disabling auth requirements
        test_config = self.temp_config.copy()
        test_config['security']['require_auth_for_network_control'] = False

        with open('config.yaml', 'w') as f:
            yaml.safe_dump(test_config, f)

        # Network control should work without auth now
        tool = NetworkControlTool()
        result = await tool.execute("enable_network", "richard_elliot")
        assert result["success"], "Network control should work without auth when disabled"

        # Restore auth requirement
        test_config['security']['require_auth_for_network_control'] = True
        with open('config.yaml', 'w') as f:
            yaml.safe_dump(test_config, f)

        print("✅ Security configuration tests passed")

    async def run_all_tests(self):
        """Run all authentication tests"""
        try:
            await self.setup_test_environment()

            print("🚀 Starting WitsV3 Authentication System Tests")
            print("=" * 60)

            await self.test_token_generation_and_hashing()
            await self.test_token_verification()
            await self.test_user_and_token_verification()
            await self.test_network_control_with_auth()
            await self.test_ethics_override_with_auth()
            await self.test_security_configuration()

            print("\n" + "=" * 60)
            print("🎉 ALL AUTHENTICATION TESTS PASSED!")
            print("✅ Token generation and hashing: PASS")
            print("✅ Token verification: PASS")
            print("✅ User and token verification: PASS")
            print("✅ Network control authentication: PASS")
            print("✅ Ethics override authentication: PASS")
            print("✅ Security configuration: PASS")
            print("\n🔐 Authentication system is secure and functional!")

            return True

        except Exception as e:
            print(f"\n❌ TEST FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False

        finally:
            await self.cleanup_test_environment()

async def main():
    """Main test function"""
    test_suite = TestAuthenticationSystem()
    success = await test_suite.run_all_tests()
    return 0 if success else 1

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(result)
    except KeyboardInterrupt:
        print("\n⛔ Tests cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        sys.exit(1)
