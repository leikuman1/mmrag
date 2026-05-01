import unittest
from urllib import error
from unittest.mock import patch

from mmrag.config import ConfigurationError, EndpointConfig
from mmrag.model_providers.base import ChatMessage
from mmrag.model_providers.openai_compatible import OpenAICompatibleChatProvider


class ModelProviderTests(unittest.TestCase):
    def test_openai_compatible_provider_wraps_network_errors(self) -> None:
        provider = OpenAICompatibleChatProvider(
            EndpointConfig(
                base_url="https://example.com/v1",
                api_key="secret",
                model="demo-model",
            )
        )
        with patch("mmrag.model_providers.openai_compatible.request.urlopen", side_effect=error.URLError("unreachable")):
            with self.assertRaises(ConfigurationError) as ctx:
                provider.generate([ChatMessage(role="user", content="hello")])
        self.assertIn("Could not reach model endpoint", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
