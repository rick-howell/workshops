import random

# ASCII art for card suits
SUITS = {
    'Hearts':    'H',
    'Diamonds':  'D',
    'Clubs':     'C',
    'Spades':    'S'
}

# Card values
VALUES = {
    'Ace': 11, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
    'Jack': 10, 'Queen': 10, 'King': 10
}

def create_deck():
    # TODO: Create and return a list of tuples representing cards
    # Each tuple should contain (value, suit)
    pass

def deal_card(deck):
    # TODO: Remove and return a random card from the deck
    pass

def calculate_hand_value(hand):
    # TODO: Calculate and return the value of a hand
    # Remember to handle Aces (1 or 11)
    pass

def display_card(card):
    # TODO: Return a string representation of a card
    # For now, you can return a simple string like "Ace of Hearts"
    pass

def display_hand(hand, hide_first=False):
    # TODO: Return a string representation of a hand
    # For now, you can return a simple string like "Ace of Hearts, King of Spades"
    return ", ".join([display_card(card) for card in hand])

def play_blackjack():
    # Create the deck
    deck = create_deck()
    
    # Shuffle, and deal initial hands to player and dealer
    player_hand = [deal_card(deck), deal_card(deck)]
    dealer_hand = [deal_card(deck), deal_card(deck)]

    # TODO: Implement the main game loop
    # 1. Player's turn: hit or stand
    player_value = calculate_hand_value(player_hand)

    '''We'll implement a while loop to allow the player to hit or stand.'''
    # ======================================================================

    # while player_value < 21:



    if player_value > 21:
        print('\nPlayer Busted! Dealer wins.\n')
        return

    # ======================================================================

    # 2. Dealer's turn (if player hasn't busted)
    # Nothing to do here until step 3.
    if player_value <= 21:
        print('\nPlayer Turn Has Ended.\n')
        # Reveal dealer's hidden card
        print('Dealer hand:')
        print(display_hand(dealer_hand))

        # Calculate the value of the dealer's hand
        dealer_value = calculate_hand_value(dealer_hand)

        # Dealer hits until hand value is 17 or higher
        while dealer_value < 17:
            dealer_hand.append(deal_card(deck))
            dealer_value = calculate_hand_value(dealer_hand)

        # Display dealer's final hand
        print('Dealer hand:')
        print(display_hand(dealer_hand))
        
        
        # 3. Determine winner and display results
        # Check if dealer has busted

        # Compare player's and dealer's hand values


    # Game over
    print('Game over.')

if __name__ == "__main__":
    play_blackjack()