#!/usr/bin/env python3
# File name: function.py

# imports
import itertools
import logging

import config as c


def points(card, trump):
    """
    Gives back the points of the card

    :param card: card for witch the points needs to be determent
    :param trump: trump suit
    :return: integer value of the points
    """
    # points if not trump
    points = {6: 0, 7: 0, 8: 0, 9: 0, 10: 10, 11: 2, 12: 3, 13: 4, 14: 11}

    # check if trump card
    if card.suit == trump.suit:
        # if nell
        if card.value == 9:
            point = 14
        # if buur
        elif card.value == 11:
            point = 20
        else:
            point = points[card.value]
    else:
        point = points[card.value]
    return point


def strength(card, trump):
    """
    Gives back the strength of the card - the higher, the better

    :param card: card for witch the strength needs to be determent
    :param trump: trump suit
    :return: integer value of the strength
    """
    if card.suit == trump.suit:
        # if nell
        if card.value == 9:
            strength = 19
        # if buur
        elif card.value == 11:
            strength = 18  # 30
        else:
            strength = card.value + 3
    else:
        strength = card.value - 6
    return strength


def card_to_number(card, trump, played):
    """
    Meta-function to easy switch to different methods
    pas-through to the methode in use

    :param card: card in question
    :param trump: trump suit
    :param played: if second player we need to know what was played ba the first player
    :return: integer value representing the card
    """

    match c.metric:
        case "card_strength":
            return strength(card, trump)
        case "linear_numbering":
            return linear_numbering(card, trump, played)
        case "digits1":
            return c.comb.index((digits1(card, trump, played)))
        case "digits2":
            return c.comb.index((digits2(card, trump, played)))
        case "simple":
            return simple(card, trump)


def linear_numbering(card, trump, played):
    """
    Convert a cart to a number value for the reinforcement learning algorithms

    :param card: card in question
    :param trump: trump suit
    :param played: if second player we need to know what was played ba the first player
    :return: integer value representing the card
    """

    s, v = str(card).split("-")
    t, _ = str(trump).split("-")
    if played == None:
        p = None
    else:
        p, _ = str(played).split("-")

    if s == t:
        # trumpcard
        match v:
            case "6":
                return 18
            case "7":
                return 19
            case "8":
                return 20
            case "9":
                return 25
            case "Banner":
                return 21
            case "Under":
                return 26
            case "Ober":
                return 22
            case "König":
                return 23
            case "Ass":
                return 24
    elif s == p:
        match v:
            case "6":
                return 9
            case "7":
                return 10
            case "8":
                return 11
            case "9":
                return 12
            case "Banner":
                return 13
            case "Under":
                return 14
            case "Ober":
                return 15
            case "König":
                return 16
            case "Ass":
                return 17
    else:
        match v:
            case "6":
                return 0
            case "7":
                return 1
            case "8":
                return 2
            case "9":
                return 3
            case "Banner":
                return 4
            case "Under":
                return 5
            case "Ober":
                return 6
            case "König":
                return 7
            case "Ass":
                return 8

    return 0


def digits1(card, trump, played):
    """
    Convert a cart to a number value for the reinforcement learning algorithms

    :param card: card in question
    :param trump: trump suit
    :param played: if second player we need to know what was played ba the first player
    :return: integer value representing the card
    """

    s, v = str(card).split("-")
    t, _ = str(trump).split("-")
    if played is None:
        p = None
    else:
        p, _ = str(played).split("-")

    """
    first digit:
    ------------
    1 if Rose
    2 if Eichel
    3 if Schellen
    4 if Schilten    
    
    second digit:
    -------------
    0 if card 6
    1 if card 7
    2 if card 8
    3 if card 9
    4 if card Banner
    5 if card Under
    6 if card Ober
    7 if card König
    8 if card Ass
    
    third digit: 
    ------------
    0 if card suit an played card suit doesn't march
    1 if played card is None (we play first)
    2 if card suit and played suit match
    3 if card suit is trump suit and trump is played 
    4 if card suit is trump suit
    
    
    Example:
    --------
    Trump: Schellen
    Card: Rosen-8
    Played: Eichel-Under
    card value: 021
    
    """
    first = "-"
    second = "-"
    third = "-"

    # match first digit
    match s:
        case "Rose":
            first = "1"
        case "Eichel":
            first = "2"
        case "Schellen":
            first = "3"
        case "Schilten":
            first = "4"

    # match second digit
    match v:
        case "6":
            second = "0"
        case "7":
            second = "1"
        case "8":
            second = "2"
        case "9":
            second = "3"
        case "Banner":
            second = "4"
        case "Under":
            second = "5"
        case "Ober":
            second = "6"
        case "König":
            second = "7"
        case "Ass":
            second = "8"

    # match third digit
    if s == t:
        third = "3"  # "4"
    # elif s == t and s == p:
    #     third = "3"
    elif s == p:
        third = "2"
    elif p is None:
        third = " 1"
    elif s != p and p is not None:
        third = "0"

    return int(f"{third}{second}{first}")


def digits2(card, trump, played):
    """
    Convert a cart to a number value for the reinforcement learning algorithms

    :param card: card in question
    :param trump: trump suit
    :param played: if second player we need to know what was played ba the first player
    :return: integer value representing the card
    """

    s, v = str(card).split("-")
    t, _ = str(trump).split("-")
    if played is None:
        p = None
    else:
        p, _ = str(played).split("-")

    """
    first digit:
    ------------
    1 if Rose
    2 if Eichel
    3 if Schellen
    4 if Schilten    

    second digit:
    -------------
    0 if card 6
    1 if card 7
    2 if card 8
    3 if card 9
    4 if card Banner
    5 if card Under
    6 if card Ober
    7 if card König
    8 if card Ass

    third digit: 
    ------------
    0 if we play first
    1 if no trump was played
    2 if trump was played
    
    fourth digit:
    -------------
    0 if player 2 takes the trick
    1 if take the trick
    2 if play suit
    3 if discard
    4 if we played first
    """

    first = "-"
    second = "-"
    third = "-"
    fourth = "-"

    # match first digit
    match s:
        case "Rose":
            first = "1"
        case "Eichel":
            first = "2"
        case "Schellen":
            first = "3"
        case "Schilten":
            first = "4"

    # match second digit
    match v:
        case "6":
            second = "0"
        case "7":
            second = "1"
        case "8":
            second = "2"
        case "9":
            second = "3"
        case "Banner":
            second = "4"
        case "Under":
            second = "5"
        case "Ober":
            second = "6"
        case "König":
            second = "7"
        case "Ass":
            second = "8"

    # match third digit
    if p is None:
        third = "0"
    elif s != t:
        third = "1"
    elif s == t:
        third = "2"

    # match fourth digit
    if p is None:
        fourth = "4"
    elif s != p and strength(card, trump) <= strength(played, trump):
        fourth = "3"
    elif s == p and strength(card, trump) < strength(played, trump):
        fourth = "2"
    elif s != p and strength(card, trump) > strength(played, trump):
        fourth = "1"
    elif s == p and strength(card, trump) > strength(played, trump):
        fourth = "1"
    elif s == p and strength(card, trump) < strength(played, trump):
        fourth = "0"
    else:
        print("Trump: ", str(trump))
        print("Card: ", str(card), " - ", strength(card, trump))
        print("Played: ", str(played), " - ", strength(played, trump))

    # return combinations(int(third + second + first))
    return int(f"{fourth}{third}{second}{first}")


def simple(card, trump):
    look_up = {'Rose-6': 1, 'Rose-7': 2, 'Rose-8': 3, 'Rose-9': 4,
               'Rose-Banner': 5, 'Rose-Under': 6, 'Rose-Ober': 7,
               'Rose-König': 8, 'Rose-Ass': 9, 'Eichel-6': 10,
               'Eichel-7': 11,
               'Eichel-8': 12, 'Eichel-9': 13, 'Eichel-Banner': 14,
               'Eichel-Under': 15, 'Eichel-Ober': 16, 'Eichel-König': 17,
               'Eichel-Ass': 18, 'Schellen-6': 19, 'Schellen-7': 20,
               'Schellen-8': 21, 'Schellen-9': 22, 'Schellen-Banner': 23,
               'Schellen-Under': 24, 'Schellen-Ober': 25,
               'Schellen-König': 26,
               'Schellen-Ass': 27, 'Schilten-6': 28, 'Schilten-7': 29,
               'Schilten-8': 30, 'Schilten-9': 31, 'Schilten-Banner': 32,
               'Schilten-Under': 33, 'Schilten-Ober': 34,
               'Schilten-König': 35,
               'Schilten-Ass': 36, 'Trump-6': 37, 'Trump-7': 38,
               'Trump-8': 39,
               'Trump-9': 40, 'Trump-Banner': 41, 'Trump-Under': 42,
               'Trump-Ober': 43, 'Trump-König': 44, 'Trump-Ass': 45}
    if card.suit == trump.suit:
        foo = str(card).split("-")
        c = f"Trump-{foo[-1]}"
        return look_up[c]
    else:
        return look_up[str(card)]


def combinations(value):
    """
    Reduce the max number space, turn number to number index
    We skip the gaps in the card number space and reduce memory and compute
    time

    :param value: card number computed from the card to number method
    :return: index of the number in the combination list
    """
    # Don't run this part all the time, it takes too much time
    # do it once and save the output for later use
    s = ["1", "2", "3", "4"]
    v = ["0", "1", "2", "3", "4", "5", "6", "7", "8"]
    pl = ["0", "1", "2"]
    st = ["0", "1", "2", "3", "4"]

    comb = []
    for item in list(itertools.product(st, pl, v, s)):
        comb.append(int(f"{item[0]}{item[1]}{item[2]}{item[3]}"))
    print(comb)
    print(len(comb))

    idx = c.comb.index(value)
    # print(idx)
    return idx


def number_to_card(value):
    """

    Doesn't work with card strength and linear numbering

    :param value:
    :return:
    """
    match c.num_to_card:
        case "digits1":
            return re_digits1(value)
        case "digits2":
            return re_digits2(value)
        case "re_simple":
            return re_simple(value)
        case _:
            logging.critical(f"Wrong or missing number to card choice ({c.d})")

def re_simple(value):
    look_up = {'Rose-6': 1, 'Rose-7': 2, 'Rose-8': 3, 'Rose-9': 4,
               'Rose-Banner': 5, 'Rose-Under': 6, 'Rose-Ober': 7,
               'Rose-König': 8, 'Rose-Ass': 9, 'Eichel-6': 10,
               'Eichel-7': 11,
               'Eichel-8': 12, 'Eichel-9': 13, 'Eichel-Banner': 14,
               'Eichel-Under': 15, 'Eichel-Ober': 16, 'Eichel-König': 17,
               'Eichel-Ass': 18, 'Schellen-6': 19, 'Schellen-7': 20,
               'Schellen-8': 21, 'Schellen-9': 22, 'Schellen-Banner': 23,
               'Schellen-Under': 24, 'Schellen-Ober': 25,
               'Schellen-König': 26,
               'Schellen-Ass': 27, 'Schilten-6': 28, 'Schilten-7': 29,
               'Schilten-8': 30, 'Schilten-9': 31, 'Schilten-Banner': 32,
               'Schilten-Under': 33, 'Schilten-Ober': 34,
               'Schilten-König': 35,
               'Schilten-Ass': 36, 'Trump-6': 37, 'Trump-7': 38,
               'Trump-8': 39,
               'Trump-9': 40, 'Trump-Banner': 41, 'Trump-Under': 42,
               'Trump-Ober': 43, 'Trump-König': 44, 'Trump-Ass': 45}
    str_card = {i for i in look_up if look_up[i] == value}
    for item in str_card:
        str_card = item
    suit, value = str_card.split("-")

    suite_list = ["Rose", "Eichel", "Schellen", "Schilten"]
    value_list = [None, None, None, None, None, None, "6", "7", "8", "9", "Banner",
                "Under", "Ober", "König", "Ass"]
    print(suit, value)
    s = suite_list.index(str(suit))
    v = value_list.index((str(value)))

    return s, v



def re_digits1(value):
    if len(str(value)) == 1:
        s = value - 1
        v = 6
    else:
        s, v = int(str(value)[-1]) - 1, int(str(value)[-2]) + 6
    # print("v, s: ", v, s)

    return s, v


def re_digits2(value):
    s, v = (int(value[-1]) - 1, int(value[-2]) + 6)
    return s, v


def cardlist_to_text(card_list):
    return list(map(str, card_list))


def main():
    """
    Tests for the file, does it run?
    """

    import env
    deck = env.Deck()
    card = deck.cards[0]
    played = deck.cards[1]
    trump = deck.cards[-1]

    r = card_to_number(card, card, None)

    # print(f"Card: {card}")
    # print(f"Trump: {trump}")
    # print(f"Played: {played}")
    # print(f"Value: {r}")

    return r


if __name__ == '__main__':
    s = []
    for i in range(0, 1):
        pass
        # s.append(main())
        # print(combinations(32))

    # print(max(s))
    re_simple(12)

