from argparse import ArgumentParser


def get_args():
    """ Parses the command line arguments and returns them. """
    parser = ArgumentParser(prog='tennis_gym',
                            description='Gym environment for Pong game.')

    # Argument for the mode of execution (human, random or robot):
    parser.add_argument(
        "player1",
        type=str,
        nargs='?',
        default="human",
        choices=["human", "god", "robot"],
        help="First player type.",
    )

    parser.add_argument(
        "player2",
        type=str,
        nargs='?',
        default="human",
        choices=["human", "god", "robot"],
        help="Second player type."
    )

    return parser.parse_args()
