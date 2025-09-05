#!/usr/bin/env python3
"""
Script để gửi fake webhook đến FastAPI endpoint
"""

import requests
import json
from datetime import datetime, timezone
import uuid
import random

API_URL = "http://localhost:8000/api/v1/webhooks/github"


def generate_fake_commit():
    """Tạo fake commit data"""
    commit_hash = ''.join(random.choices('abcdef0123456789', k=40))

    fake_messages = [
        "Fix bug in user authentication #123",
        "Add new feature for dashboard closes #456",
        "Update documentation fixes #789",
        "Refactor code and improve performance",
        "Fix memory leak in data processing #234",
        "Add unit tests for payment module",
        "Update dependencies and security patches",
        "Implement new API endpoint resolves #567"
    ]

    fake_authors = [
        {"name": "John Doe", "email": "john.doe@company.com"},
        {"name": "Jane Smith", "email": "jane.smith@company.com"},
        {"name": "Bob Wilson", "email": "bob.wilson@company.com"},
        {"name": "Alice Brown", "email": "alice.brown@company.com"}
    ]

    fake_files = {
        "added": [
            "src/new_feature.py",
            "tests/test_new_feature.py",
            "docs/api.md",
            "config/settings.json"
        ],
        "modified": [
            "src/main.py",
            "src/utils/helpers.py",
            "README.md",
            "requirements.txt",
            "src/auth/login.py",
            "src/models/user.py"
        ],
        "removed": [
            "src/old_file.py",
            "deprecated/old_module.py",
            "temp/debug.log"
        ]
    }

    author = random.choice(fake_authors)
    message = random.choice(fake_messages)

    return {
        "id": commit_hash,
        "message": message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "author": {
            "name": author["name"],
            "email": author["email"],
            "username": author["name"].lower().replace(" ", "")
        },
        "committer": {
            "name": author["name"],
            "email": author["email"],
            "username": author["name"].lower().replace(" ", "")
        },
        "added": random.sample(fake_files["added"], random.randint(0, 2)),
        "modified": random.sample(fake_files["modified"], random.randint(1, 3)),
        "removed": random.sample(fake_files["removed"], random.randint(0, 1)),
        "url": f"https://github.com/company/project/commit/{commit_hash}"
    }


def create_fake_webhook_payload(num_commits=2, repo_name="test-project"):
    """Tạo fake webhook payload"""

    commits = []
    for _ in range(num_commits):
        commits.append(generate_fake_commit())

    # Head commit là commit cuối cùng
    head_commit = commits[-1] if commits else generate_fake_commit()

    payload = {
        "ref": "refs/heads/main",
        "repository": {
            "name": repo_name,
            "full_name": f"company/{repo_name}",
            "description": f"Test repository: {repo_name}",
            "private": False,
            "html_url": f"https://github.com/company/{repo_name}"
        },
        "commits": commits,
        "head_commit": head_commit,
        "pusher": {
            "name": head_commit["author"]["name"],
            "email": head_commit["author"]["email"]
        }
    }

    return payload


def send_webhook(payload, signature=None):
    """Gửi webhook đến API"""
    headers = {
        'Content-Type': 'application/json',
        'User-Agent': 'GitHub-Hookshot/abc123'
    }

    # Thêm signature nếu có (để bypass verify_webhook_signature)
    if signature:
        headers['X-Hub-Signature-256'] = signature

    try:
        print(f"🔄 Sending webhook to {API_URL}...")
        print(f"📦 Payload: {len(payload['commits'])} commits in {payload['repository']['name']}")

        response = requests.post(
            API_URL,
            json=payload,
            headers=headers,
            timeout=30
        )

        print(f"📊 Status Code: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("✅ Webhook sent successfully!")
            print(f"📝 Response: {json.dumps(result, indent=2)}")
            return True
        else:
            print(f"❌ Webhook failed: {response.text}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"❌ Error sending webhook: {e}")
        return False


def main():
    """Main function"""
    print("🚀 Fake GitHub Webhook Sender")
    print("-" * 40)

    # Tạo menu
    print("Choose an option:")
    print("1. Send single commit")
    print("2. Send multiple commits (2-5)")
    print("3. Send custom payload")
    print("4. Send batch of webhooks")

    choice = input("\nEnter choice (1-4): ").strip()

    if choice == "1":
        # Single commit
        payload = create_fake_webhook_payload(num_commits=1)
        send_webhook(payload)

    elif choice == "2":
        # Multiple commits
        num_commits = random.randint(2, 5)
        payload = create_fake_webhook_payload(num_commits=num_commits)
        send_webhook(payload)

    elif choice == "3":
        # Custom payload
        repo_name = input("Repository name (default: test-project): ").strip() or "test-project"
        num_commits = int(input("Number of commits (default: 2): ").strip() or "2")
        payload = create_fake_webhook_payload(num_commits=num_commits, repo_name=repo_name)
        send_webhook(payload)

    elif choice == "4":
        # Batch webhooks
        num_webhooks = int(input("Number of webhooks to send (default: 5): ").strip() or "5")
        success_count = 0

        for i in range(num_webhooks):
            print(f"\n📤 Sending webhook {i + 1}/{num_webhooks}")
            payload = create_fake_webhook_payload(
                num_commits=random.randint(1, 3),
                repo_name=f"project-{i + 1}"
            )
            if send_webhook(payload):
                success_count += 1

        print(f"\n📊 Summary: {success_count}/{num_webhooks} webhooks sent successfully")

    else:
        print("❌ Invalid choice")


if __name__ == "__main__":
    main()
