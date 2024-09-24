import unittest
from blackjack import create_deck, deal_card, calculate_hand_value, display_card, display_hand, SUITS, VALUES

class TestBlackjack(unittest.TestCase):

    def test_create_deck(self):
        deck = create_deck()
        self.assertEqual(len(deck), 52)  # A standard deck has 52 cards
        self.assertTrue(all(card[0] in VALUES and card[1] in SUITS for card in deck))

    def test_deal_card(self):
        deck = create_deck()
        initial_length = len(deck)
        card = deal_card(deck)
        self.assertEqual(len(deck), initial_length - 1)
        self.assertIn(card[0], VALUES)
        self.assertIn(card[1], SUITS)

    def test_calculate_hand_value(self):
        # Test regular hand
        hand = [('10', 'Hearts'), ('King', 'Spades')]
        self.assertEqual(calculate_hand_value(hand), 20)

        # Test hand with Ace
        hand = [('Ace', 'Hearts'), ('5', 'Spades')]
        self.assertEqual(calculate_hand_value(hand), 16)

        # Test hand with Ace that should be 1
        hand = [('Ace', 'Hearts'), ('King', 'Spades'), ('5', 'Clubs')]
        self.assertEqual(calculate_hand_value(hand), 16)

    def test_display_card(self):
        card = ('Ace', 'Hearts')
        display = display_card(card)
        self.assertIn('A', display)
        self.assertIn(SUITS.get('Hearts'), display)

    def test_display_hand(self):
        hand = [('Ace', 'Hearts'), ('King', 'Spades')]
        
        # Test visible hand
        visible_display = display_hand(hand)
        self.assertIn('A', visible_display)
        self.assertIn('K', visible_display)
        self.assertIn(SUITS.get('Hearts'), visible_display)
        self.assertIn(SUITS.get('Spades'), visible_display)

        # Test hand with hidden card
        hidden_display = display_hand(hand, hide_first=True)
        self.assertNotIn('A', hidden_display)
        self.assertNotIn(SUITS.get('Hearts'), hidden_display)
        self.assertIn('K', hidden_display)
        self.assertIn(SUITS.get('Spades'), hidden_display)

if __name__ == '__main__':
    unittest.main(verbosity=2)