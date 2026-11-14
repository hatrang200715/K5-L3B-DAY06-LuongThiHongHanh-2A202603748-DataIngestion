import unittest
from unittest.mock import patch

from src.chatbot import RetrievalChatbot


class FakeEmbedder:
    provider = "openrouter"

    def embed_query(self, question: str) -> list[float]:
        return [0.1, 0.2]


class FakeStore:
    def __init__(self, hits: list[dict] | None = None) -> None:
        self.hits = hits or []
        self.last_search: tuple | None = None

    def search(self, strategy, vector, top_k, filters):
        self.last_search = (strategy, vector, top_k, filters)
        return self.hits


class RetrievalChatbotTests(unittest.TestCase):
    def test_strategy_c_passes_only_the_selected_section_filter(self) -> None:
        store = FakeStore([{"page": 17, "text": "Evidence", "metadata": {}}])
        chatbot = RetrievalChatbot(FakeEmbedder(), store, chat_model="test-model")

        with patch.object(chatbot, "_generate_answer", return_value="Grounded [1]"):
            result = chatbot.answer("A question", "C", section="introduction")

        self.assertEqual(store.last_search, ("C", [0.1, 0.2], 3, {"section": "introduction"}))
        self.assertEqual(result["answer"], "Grounded [1]")
        self.assertEqual(len(result["hits"]), 1)

    def test_strategy_a_does_not_apply_metadata_filter(self) -> None:
        store = FakeStore([{"page": 17, "text": "Evidence", "metadata": {}}])
        chatbot = RetrievalChatbot(FakeEmbedder(), store, chat_model="test-model")

        with patch.object(chatbot, "_generate_answer", return_value="Grounded [1]"):
            chatbot.answer("A question", "A", section="introduction")

        self.assertEqual(store.last_search, ("A", [0.1, 0.2], 3, None))

    def test_no_hits_returns_clear_message_without_calling_chat_model(self) -> None:
        chatbot = RetrievalChatbot(FakeEmbedder(), FakeStore(), chat_model="test-model")

        with patch.object(chatbot, "_generate_answer") as generate:
            result = chatbot.answer("A question", "B")

        self.assertIn("Không tìm thấy", result["answer"])
        self.assertEqual(result["hits"], [])
        generate.assert_not_called()

    def test_missing_chat_model_is_an_explicit_configuration_error(self) -> None:
        with patch.dict("os.environ", {"CHAT_MODEL": ""}):
            with self.assertRaisesRegex(ValueError, "Set CHAT_MODEL"):
                RetrievalChatbot(FakeEmbedder(), FakeStore())

    def test_openrouter_response_is_parsed_without_leaking_credentials(self) -> None:
        chatbot = RetrievalChatbot(FakeEmbedder(), FakeStore(), chat_model="test-model")
        response = {"choices": [{"message": {"content": "Câu trả lời [1]"}}]}

        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            with patch.object(chatbot, "_post_json", return_value=response) as post_json:
                answer = chatbot._generate_openai_compatible("system", "question")

        self.assertEqual(answer, "Câu trả lời [1]")
        self.assertEqual(
            post_json.call_args.args[1]["Authorization"],
            "Bearer test-key",
        )


if __name__ == "__main__":
    unittest.main()
