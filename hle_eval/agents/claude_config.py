"""Configuration for Claude Code CLI agent using Anthropic credentials."""
import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


class ClaudeConfig:
    """Configuration for Claude Code agent with Anthropic API credentials."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "haiku",
        timeout: float = 1200.0,
    ):
        """
        Initialize Claude Code configuration.

        Args:
            api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var
            base_url: Anthropic base URL. If None, reads from ANTHROPIC_BASE_URL env var
            model: Model name to use (default: sonnet)
            timeout: Timeout in seconds for CLI execution (default: 900)
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.base_url = base_url or os.environ.get("ANTHROPIC_BASE_URL")
        self.model = model
        self.timeout = timeout

        # Validate API key
        if not self.api_key:
            raise EnvironmentError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )

    def get_env_dict(self) -> dict[str, str]:
        """
        Get environment variables dictionary for subprocess execution.

        Returns:
            Dictionary with Anthropic credentials for subprocess environment
        """
        env = os.environ.copy()
        env["ANTHROPIC_API_KEY"] = self.api_key

        if self.base_url:
            env["ANTHROPIC_BASE_URL"] = self.base_url

        # Remove Claude Code session vars to allow nested CLI invocations
        for key in ["CLAUDECODE", "CLAUDE_CODE_SSE_PORT", "CLAUDE_CODE_ENTRYPOINT"]:
            env.pop(key, None)

        return env

    def __repr__(self) -> str:
        """Return string representation of config."""
        masked_key = f"{self.api_key[:10]}..." if self.api_key else "Not set"
        return (
            f"ClaudeConfig(model={self.model}, "
            f"api_key={masked_key}, "
            f"base_url={self.base_url}, "
            f"timeout={self.timeout})"
        )
