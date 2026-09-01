import unittest

from app.db.models import PortfolioSnapshot, RecommendationRun
from app.services.review import calculate_review


class ReviewServiceTests(unittest.TestCase):
    def test_drift_is_deterministic_and_contributions_are_prioritized(self):
        classes = [
            {"key": "brazilian_stocks", "target_weight": 0.20},
            {"key": "fiis", "target_weight": 0.20},
            {"key": "international", "target_weight": 0.20},
            {"key": "fixed_income", "target_weight": 0.30},
            {"key": "crypto", "target_weight": 0.10},
        ]
        recommendation = RecommendationRun(id="rec-1", classes=classes)
        portfolio = PortfolioSnapshot(
            id="portfolio-1",
            total_value_brl=1000,
            classes={
                "brazilian_stocks": 50,
                "fiis": 100,
                "international": 250,
                "fixed_income": 500,
                "crypto": 100,
            },
            normalized_weights={
                "brazilian_stocks": 0.05,
                "fiis": 0.10,
                "international": 0.25,
                "fixed_income": 0.50,
                "crypto": 0.10,
            },
        )

        review = calculate_review(recommendation, portfolio)

        self.assertEqual(review["drift_band"], 0.05)
        self.assertEqual(review["items"][0]["suggested_action"], "contribute")
        self.assertEqual(review["items"][0]["class_key"], "brazilian_stocks")
        self.assertEqual(review["items"][-1]["suggested_action"], "review_sale")
        self.assertEqual(review["items"][0]["value_gap_brl"], 150)
        self.assertNotIn("order", review)


if __name__ == "__main__":
    unittest.main()
