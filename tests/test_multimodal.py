from unittest.mock import MagicMock, patch

from core import Collection
from llm import Embedder


def test_embedder_image_path_detection():
    emb = Embedder("dummy")
    # Mocking os.path.exists to return True for our fake images
    with (
        patch("os.path.exists", return_value=True),
        patch("llm.Embedder._embed_images") as mock_img_embed,
    ):
        mock_img_embed.return_value = [[0.1, 0.2]]

        # This should trigger _embed_images
        res = emb.embed_texts(["test.jpg"])
        mock_img_embed.assert_called_once()
        assert res == [[0.1, 0.2]]


def test_embedder_images_functional():
    emb = Embedder("dummy")
    mock_response = {"data": [{"embedding": [0.5, 0.6]}]}

    # Mock open returning a bytes object
    mock_file = MagicMock()
    mock_file.read.return_value = b"fake_image_data"
    mock_file.__enter__.return_value = mock_file

    with (
        patch("builtins.open", return_value=mock_file),
        patch("litellm.embedding", return_value=mock_response),
    ):
        res = emb._embed_images(["fake.png"])
        assert res == [[0.5, 0.6]]


def test_collection_multimodal_flow():
    col = Collection(dimension=2)

    with patch("llm.Embedder.embed_texts") as mock_embed:
        mock_embed.return_value = [[0.1, 0.2], [0.3, 0.4]]

        col.add_images(["img1.jpg", "img2.jpg"])
        assert len(col._metadata) == 2
        assert col._metadata[0]["image_path"] == "img1.jpg"

    with patch("llm.Embedder.embed_query") as mock_q:
        mock_q.return_value = [0.1, 0.2]
        results = col.search_image("query.jpg", top_k=1)
        assert results[0][0]["image_path"] == "img1.jpg"
