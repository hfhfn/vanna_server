from types import SimpleNamespace
from typing import Any, Union, List
import os
import requests


def convert_to_namespace(obj: Any) -> Any:
    """Recursively convert a dictionary or list to a SimpleNamespace object."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: convert_to_namespace(v) for k, v in obj.items()})
    elif isinstance(obj, list):
        return [convert_to_namespace(item) for item in obj]
    else:
        return obj


class CustomEmbedding:
    """A client for the Siliconflow Embeddings API with chainable call style.

    Example usage:
        >>> sf = CustomEmbedding(api_key="your_api_key")
        >>> result = sf.embeddings.create(model="your_model", input="your_text")
    """

    def __init__(self, base_url: str, api_key: str):
        """Initialize the embedding client.

        Args:
            api_key: Your Siliconflow API key
        """
        self.base_url = base_url
        self.api_key = api_key
        self.embeddings = self._Embeddings(self)

    class _Embeddings:
        """Inner class handling embeddings operations."""

        def __init__(self, parent):
            self._parent = parent
            self._url = os.path.join(self._parent.base_url, "embeddings")
            self._headers = {
                "Authorization": f"Bearer {self._parent.api_key}",
                "Content-Type": "application/json"
            }

        def create(
                self,
                model: str,
                input: Union[str, List[str]],
                encoding_format: str = "float"
        ) -> SimpleNamespace:
            """Create embeddings for the input text(s).

            Args:
                model: The model ID to use for embedding
                input: Input text or list of texts to embed
                encoding_format: The format of the output embeddings

            Returns:
                API response as a dictionary
            """
            if not input:
                raise ValueError("Input cannot be empty")

            response = requests.post(
                self._url,
                json={
                    "model": model,
                    "input": input,
                    "encoding_format": encoding_format
                },
                headers=self._headers
            )

            response.raise_for_status()
            return convert_to_namespace(response.json())


# 使用示例
if __name__ == "__main__":
    # 初始化客户端
    client = CustomEmbedding(base_url="", api_key="<your_api_key>")

    # 调用聊天接口 - 新风格
    response = client.embeddings.create(model="your_model", input="your_text")

    print(response)