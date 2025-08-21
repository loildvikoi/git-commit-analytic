from typing import List
from .....domain.entities.commit import Commit


class CommitPrompts:
    """Enhanced prompts for commit analysis with better AI engineering"""

    def get_commit_analysis_prompt(self, commit: Commit) -> str:
        """Generate detailed prompt for analyzing a single commit"""

        # Build file changes summary
        files_summary = self._build_files_summary(commit.files_changed)

        # Build context information
        context_info = self._build_context_info(commit)

        return f"""
You are an expert software engineer analyzing Git commits. Your task is to provide a comprehensive analysis of the following commit.

COMMIT INFORMATION:
- Hash: {commit.commit_hash.value}
- Author: {commit.author_name} <{commit.author_email}>
- Date: {commit.timestamp.isoformat()}
- Branch: {commit.branch}
- Project: {commit.project}
- Message: "{commit.message}"

{context_info}

FILE CHANGES:
{files_summary}

ANALYSIS REQUIREMENTS:
Analyze this commit and provide insights on:

1. **Summary**: A concise 1-2 sentence summary of what this commit does
2. **Type Classification**: Categorize the commit type(s)
3. **Sentiment Analysis**: Assess the nature of changes (-1.0 to 1.0 scale)
4. **Confidence**: Your confidence in this analysis (0.0 to 1.0)
5. **Entities**: Extract important technical entities (files, functions, issues, etc.)

CLASSIFICATION GUIDELINES:
- **feature**: New functionality or capabilities
- **bugfix**: Fixes bugs or issues
- **refactor**: Code restructuring without changing functionality
- **docs**: Documentation changes
- **test**: Test-related changes
- **config**: Configuration or build changes
- **security**: Security-related improvements
- **performance**: Performance optimizations
- **hotfix**: Critical emergency fixes
- **chore**: Maintenance tasks

SENTIMENT SCALE:
- 1.0: Major positive feature addition
- 0.5: Positive improvement or enhancement
- 0.0: Neutral change (refactor, docs)
- -0.3: Bug fix (negative but necessary)
- -0.7: Critical bug fix or security issue
- -1.0: Major breaking change or critical failure

OUTPUT FORMAT:
Respond with valid JSON only, no additional text:

{{
  "summary": "Clear description of what this commit accomplishes",
  "tags": ["primary_type", "secondary_type", "optional_modifier"],
  "sentiment": 0.0,
  "confidence": 0.85,
  "entities": ["important_file.py", "function_name", "#123", "component_name"]
}}

EXAMPLE:
{{
  "summary": "Implement user authentication system with JWT tokens and password hashing",
  "tags": ["feature", "security", "authentication"],
  "sentiment": 0.7,
  "confidence": 0.9,
  "entities": ["auth.py", "User", "JWT", "bcrypt", "login_endpoint"]
}}

Now analyze the commit:
        """

    def get_batch_summary_prompt(self, commits: List[Commit]) -> str:
        """Generate prompt for summarizing multiple commits"""
        if not commits:
            return "No commits to analyze."

        commits_info = []
        for i, commit in enumerate(commits[:10], 1):  # Limit to 10 commits
            files_count = len(commit.files_changed)
            commits_info.append(
                f"{i}. [{commit.commit_hash.value[:8]}] {commit.message} "
                f"by {commit.author_name} ({files_count} files)"
            )

        commits_text = "\n".join(commits_info)

        return f"""
You are analyzing a series of Git commits to provide a comprehensive summary.

COMMITS TO ANALYZE ({len(commits)} total):
{commits_text}

TASK:
Provide a well-structured summary that includes:

1. **Overview**: High-level summary of what was accomplished
2. **Key Changes**: Most significant modifications
3. **Contributors**: Main contributors and their focus areas
4. **Patterns**: Common themes or development patterns
5. **Impact**: Likely impact on the codebase/project

GUIDELINES:
- Focus on business value and technical impact
- Group related changes together
- Highlight any critical fixes or major features
- Mention if there are any concerning patterns
- Keep it concise but informative (3-5 paragraphs max)

Provide a natural language summary, not JSON format.
        """

    def get_question_answer_prompt(self, question: str, context_commits: List[Commit]) -> str:
        """Generate prompt for answering questions about commits"""

        # Build context from commits
        context_info = []
        for commit in context_commits[:20]:  # Limit context
            summary = f"[{commit.timestamp.isoformat()}][Project: {commit.project}][Branch: {commit.branch}][Author name: {commit.author_name}][Author email: {commit.author_email}][Commit ID: {commit.commit_hash.value[:8]}] {commit.message}"
            if commit.summary:
                summary += f" | AI Summary: {commit.summary}"
            context_info.append(summary)

        context_text = "\n".join(context_info)

        return f"""
You are an AI assistant helping developers understand their Git commit history. 
Answer the user's question based on the provided commit context.

USER QUESTION:
{question}

COMMIT CONTEXT ({len(context_commits)} commits):
{context_text}

INSTRUCTIONS:
- Answer directly and specifically based on the commit data
- If the question cannot be answered from the context, say so clearly
- Provide specific examples with commit hashes when relevant
- Be helpful and informative
- If asking about patterns, analyze across multiple commits
- Include relevant details like author names, dates, or file changes when helpful

RESPONSE FORMAT:
Provide a natural, conversational response. Do not use JSON format.
        """

    def get_entity_extraction_prompt(self, text: str) -> str:
        """Generate prompt for extracting entities from commit-related text"""
        return f"""
Extract important technical entities from the following text related to software development.

TEXT TO ANALYZE:
{text}

ENTITY TYPES TO EXTRACT:
- File names and paths
- Function/method names
- Class names
- Variable names
- Issue/ticket numbers (e.g., #123, JIRA-456)
- Technology names (frameworks, libraries, languages)
- Component/module names
- Configuration keys
- Database table/column names
- API endpoints

GUIDELINES:
- Only extract entities that appear to be technical/development-related
- Include file extensions when present
- Preserve exact casing for code-related entities
- Include issue numbers with their prefixes
- Avoid common words unless they're clearly technical terms

OUTPUT FORMAT:
{{
  "entities": ["entity1", "entity2", "entity3"]
}}

Respond with valid JSON only.
        """

    def _build_files_summary(self, files_changed) -> str:
        """Build a summary of file changes"""
        if not files_changed:
            return "No file changes recorded."

        summary_lines = []
        total_additions = 0
        total_deletions = 0

        for file_change in files_changed:
            status_symbol = {
                'added': '+',
                'modified': '~',
                'deleted': '-',
                'renamed': '→'
            }.get(file_change.status, '?')

            summary_lines.append(
                f"  {status_symbol} {file_change.filename} "
                f"(+{file_change.additions}/-{file_change.deletions})"
            )

            total_additions += file_change.additions
            total_deletions += file_change.deletions

        files_text = "\n".join(summary_lines)

        return f"""
Files changed: {len(files_changed)}
Total changes: +{total_additions}/-{total_deletions} lines

{files_text}
        """

    def _build_context_info(self, commit: Commit) -> str:
        """Build additional context information"""
        context_parts = []

        # Issue numbers
        if commit.issue_numbers:
            issues = ", ".join(commit.issue_numbers)
            context_parts.append(f"Related Issues: {issues}")

        # Commit metrics
        metrics = commit.metrics
        context_parts.append(f"Complexity Score: {metrics.complexity_score:.2f}")
        context_parts.append(f"Impact Score: {metrics.impact_score:.2f}")

        # Commit classification hints
        if commit.is_hotfix():
            context_parts.append("⚠️  Identified as potential HOTFIX")
        elif commit.is_feature():
            context_parts.append("✨ Identified as potential FEATURE")

        return "\n".join(context_parts) if context_parts else ""
