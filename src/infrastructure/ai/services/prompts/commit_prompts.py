class CommitPrompts:
    """Prompts for commit analysis"""

    def get_commit_analysis_prompt(self, commit):
        """Generate prompt for analyzing a single commit"""
        return (
            f"Analyze the following commit:\n\n"
            f"Commit Hash: {commit.commit_hash.value}\n"
            f"Author: {commit.author.name} <{commit.author.email}>\n"
            f"Date: {commit.date.isoformat()}\n"
            f"Message: {commit.message}\n\n"
            f"Please provide a detailed analysis including:\n"
            f"- Summary of changes\n"
            f"- Tags or labels (e.g., bugfix, feature, refactor)\n"
            f"- Sentiment score (0.0 to 1.0)\n"
            f"- Confidence score (0.0 to 1.0)\n"
            f"- Extracted entities (e.g., file names, issue IDs)\n\n"
            f"Respond in JSON format with the following structure:\n"
            f"{{\n"
            f'  "summary": "string",\n'
            f'  "tags": ["string"],\n'
            f'  "sentiment": float,\n'
            f'  "confidence": float,\n'
            f'  "entities": ["string"]\n'
            f"}}\n\n"
        )
