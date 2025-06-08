#!/usr/bin/env python3
"""
WitsV3 Authentication Setup Script
Generates and configures secure authentication tokens for system access
"""

import sys
import os
import getpass
from pathlib import Path

# Add the project root to the path so we can import WitsV3 modules
sys.path.insert(0, str(Path(__file__).parent))

from core.auth_manager import get_auth_manager, generate_new_token

def main():
    """Main setup function"""
    print("🔐 WitsV3 Authentication Setup")
    print("=" * 50)

    auth_manager = get_auth_manager()

    # Check if token is already configured
    if auth_manager.is_token_configured():
        print("⚠️  Authentication token is already configured.")
        choice = input("Do you want to generate a new token? (y/N): ").strip().lower()
        if choice not in ['y', 'yes']:
            print("Setup cancelled.")
            return

    print("\n🔑 Generating secure authentication token...")
    token = generate_new_token()

    print(f"\n✅ Your new authentication token is:")
    print(f"┌{'─' * 66}┐")
    print(f"│ {token:<64} │")
    print(f"└{'─' * 66}┘")

    print("\n⚠️  IMPORTANT SECURITY NOTES:")
    print("• This token grants full administrative access to WitsV3")
    print("• Store it securely - it will NOT be shown again")
    print("• Anyone with this token can override network restrictions and ethics")
    print("• The token is cryptographically hashed before storage")

    # Confirm setup
    print("\n📝 Configuring system...")
    if auth_manager.setup_initial_token(token):
        print("✅ Authentication token configured successfully!")

        print("\n🔧 Usage Examples:")
        print("1. Enable network access:")
        print(f"   network_control(action='enable_network', user_id='richard_elliot', auth_token='{token[:16]}...')")

        print("\n2. Override ethics for testing:")
        print(f"   personality_manager.enable_ethics_override('richard_elliot', 'testing', auth_token='{token[:16]}...')")

        print("\n📁 You can also store this token in:")
        print("   • Environment variable: WITSV3_AUTH_TOKEN")
        print("   • Secure file: ~/.witsv3/auth_token")
        print("   • Your password manager")

        # Ask if user wants to save to environment file
        save_env = input("\n💾 Create .env file with token? (y/N): ").strip().lower()
        if save_env in ['y', 'yes']:
            try:
                with open('.env', 'w') as f:
                    f.write(f"# WitsV3 Authentication Token\n")
                    f.write(f"# Generated: {auth_manager.config.memory_manager.get('timestamp', 'unknown')}\n")
                    f.write(f"WITSV3_AUTH_TOKEN={token}\n")
                print("✅ Token saved to .env file")
                print("⚠️  Add .env to your .gitignore to prevent accidental commits!")
            except Exception as e:
                print(f"❌ Failed to create .env file: {e}")
    else:
        print("❌ Failed to configure authentication token")
        return 1

    print("\n🎯 Authentication setup complete!")
    print("You can now use secure network control and ethics override functions.")

    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⛔ Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)
