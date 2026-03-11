"""Configuration for OpenHands CLI agent using LLM credentials."""
import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


class OpenHandsConfig:
    """Configuration for OpenHands agent with LLM API credentials."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "anthropic/claude-sonnet-4-5",
        timeout: float = 600.0,
        reasoning_effort: str = "high",
    ):
        """
        Initialize OpenHands configuration.

        Args:
            api_key: LLM API key. If None, reads from LLM_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY env var
            base_url: LLM base URL. If None, reads from LLM_BASE_URL env var
            model: Model name to use (default: anthropic/claude-sonnet-4-5)
            timeout: Timeout in seconds for CLI execution (default: 600)
            reasoning_effort: Reasoning effort level (default: high)
        """
        self.api_key = (
            api_key
            or os.environ.get("LLM_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("OPENAI_API_KEY", "")
        )
        raw_base_url = base_url or os.environ.get("LLM_BASE_URL")
        if raw_base_url:
            self.base_url = raw_base_url.rstrip("/")
            # Strip trailing /v1 only for Anthropic models — litellm appends /v1/messages.
            # OpenAI models need /v1 because the SDK appends /chat/completions directly.
            if "anthropic" in model.lower() or "claude" in model.lower():
                self.base_url = self.base_url.removesuffix("/v1")
        else:
            self.base_url = None
        self.model = model
        self.timeout = timeout
        self.reasoning_effort = reasoning_effort

        # Validate API key
        if not self.api_key:
            raise EnvironmentError(
                "LLM API key not found. Set LLM_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY "
                "environment variable or pass api_key parameter."
            )

    def get_env_dict(self) -> dict[str, str]:
        """
        Get environment variables dictionary for subprocess execution.

        Returns:
            Dictionary with LLM credentials for subprocess environment
        """
        env = os.environ.copy()
        env["LLM_API_KEY"] = self.api_key

        if self.base_url:
            env["LLM_BASE_URL"] = self.base_url

        env["LLM_REASONING_EFFORT"] = self.reasoning_effort

        # Configure non-interactive mode
        env["TTY_INTERACTIVE"] = "0"

        # Agent and sandbox settings (matching Harbor)
        env["AGENT_ENABLE_BROWSING"] = "false"
        env["ENABLE_BROWSER"] = "false"
        env["SANDBOX_ENABLE_AUTO_LINT"] = "true"
        env["AGENT_ENABLE_PROMPT_EXTENSIONS"] = "false"
        env["SKIP_DEPENDENCY_CHECK"] = "1"
        env["RUN_AS_OPENHANDS"] = "false"
        env["RUNTIME"] = "local"

        # Logging settings
        env["FILE_STORE"] = "local"
        env["LLM_LOG_COMPLETIONS"] = "true"

        # Disable OpenAI Responses API to use Chat Completions instead
        # This avoids encrypted_content handling issues with gpt-5-mini
        env["LLM_USE_OPENAI_RESPONSES"] = "false"

        return env

    def __repr__(self) -> str:
        """Return string representation of config."""
        masked_key = f"{self.api_key[:10]}..." if self.api_key else "Not set"
        return (
            f"OpenHandsConfig(model={self.model}, "
            f"api_key={masked_key}, "
            f"base_url={self.base_url}, "
            f"timeout={self.timeout}, "
            f"reasoning_effort={self.reasoning_effort})"
        )
