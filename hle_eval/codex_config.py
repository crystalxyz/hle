"""Configuration for Codex CLI agent using OpenAI credentials."""
import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


class CodexConfig:
    """Configuration for Codex agent with OpenAI API credentials."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "gpt-4o",
        timeout: float = 600.0,
    ):
        """
        Initialize Codex configuration.

        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var
            base_url: OpenAI base URL. If None, reads from OPENAI_BASE_URL env var
            model: Model name to use (default: gpt-4o)
            timeout: Timeout in seconds for CLI execution (default: 600)
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.model = model
        self.timeout = timeout

        # Validate API key
        if not self.api_key:
            raise EnvironmentError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

    def get_env_dict(self) -> dict[str, str]:
        """
        Get environment variables dictionary for subprocess execution.

        Returns:
            Dictionary with OpenAI credentials for subprocess environment
        """
        env = os.environ.copy()
        env["OPENAI_API_KEY"] = self.api_key

        if self.base_url:
            env["OPENAI_BASE_URL"] = self.base_url

        return env

    def __repr__(self) -> str:
        """Return string representation of config."""
        masked_key = f"{self.api_key[:10]}..." if self.api_key else "Not set"
        return (
            f"CodexConfig(model={self.model}, "
            f"api_key={masked_key}, "
            f"base_url={self.base_url}, "
            f"timeout={self.timeout})"
        )
