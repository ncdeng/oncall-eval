import unittest
from unittest.mock import patch

import run_bm25_baseline as B
import run_eval as R
import run_hybrid_baseline as H


class RetrievalBaselineTests(unittest.TestCase):
    def test_rank_scores_keeps_source_order_when_scores_tie(self):
        rank_scores = getattr(B, "rank_scores", None)
        self.assertIsNotNone(rank_scores, "baseline must expose one deterministic score-ranking rule")
        self.assertEqual(rank_scores([0.7, 0.7, 0.2]), [0, 1, 2])

    def test_rrf_scores_use_one_based_ranks(self):
        rrf_scores = getattr(H, "rrf_scores", None)
        self.assertIsNotNone(rrf_scores, "hybrid baseline must expose its RRF scoring rule")
        scores = rrf_scores([[7, 3], [7, 3]], k=60)
        self.assertAlmostEqual(scores[7], 2.0 / 61.0)
        self.assertAlmostEqual(scores[3], 2.0 / 62.0)

    def test_embedding_client_initialization_does_not_require_chat_config(self):
        init_embedding_client = getattr(R, "init_embedding_client", None)
        self.assertIsNotNone(
            init_embedding_client,
            "retrieval-only scripts need an initializer that does not require chat configuration",
        )
        values = {
            "SILICONFLOW_API_KEY": "test-key",
            "EMBED_MODEL": "test-embedding-model",
        }
        old_client, old_model = R.EMB_CLIENT, R.EMBED_MODEL
        try:
            with patch.object(R, "get_env", side_effect=lambda name: values.get(name, "")):
                with patch("openai.OpenAI", return_value=object()):
                    init_embedding_client()
            self.assertIsNotNone(R.EMB_CLIENT)
            self.assertEqual(R.EMBED_MODEL, "test-embedding-model")
        finally:
            R.EMB_CLIENT, R.EMBED_MODEL = old_client, old_model


if __name__ == "__main__":
    unittest.main()
