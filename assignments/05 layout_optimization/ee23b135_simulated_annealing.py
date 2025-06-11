import random
import warnings

import numpy as np

try:
    from ee23b135_kbda import KeyboardAnalyzer
    from ee23b135_layout import Layout

except ImportError as import_error:
    # Just in case the person using this didn't read the readme
    warnings.warn(
        "Did you forget to place `ee23b135_kbda.py` and `ee23b135_layout` in the current directory?",
        stacklevel=2,
    )
    raise import_error


def optimize(
    layout: Layout,
    text: str,
    iterations: int = 500,
    temperature: float = 1500,
    cooling_rate: float = 0.996,
) -> tuple[Layout, list[float]]:
    """
    Generates a layout optmized to reduce distance travelled (when typing the `text` given)
    This method uses the simulated annealing method to accomplish this :)

    Parameters
    ----------
    layout: Layout
        The reference layout.

    text: str
        The text used to analyze the keyboard.

        Every character in text must have been defined in the layout!

    Returns
    -------
    optimized_layout: Layout
        The most optimal (atleast half decent) layout for the given text

    distances: list[float]
        The distance travelled for each intermediate layout checked

    Raises
    ------
    ValueError
        - If the text contains an unrecognized character
    """
    random.seed()  # Set some random seed

    if len(layout.mapping) < 2:
        warnings.warn(
            f"The original layout only has `{len(layout.mapping)}` keys. (Is this a mistake?)"
        )
        return layout, [calculate_distance(layout, text)]

    # List that stores the intermediant distances visited
    distances: list[float] = [calculate_distance(layout, text)]
    best_layout = layout
    best_distance = distances[0]

    for _ in range(iterations):
        # Get a layout similar to the currrent
        adjacent_layout = get_neighbour(layout)

        current_distance = distances[-1]
        new_distance = calculate_distance(adjacent_layout, text)

        # I wanted more jumps initially, which is why I multiplied the exponenet by temperature
        p = (
            temperature * np.exp((current_distance - new_distance))
            if current_distance < new_distance
            else 0
        )

        # Update the best distance if required
        if new_distance < best_distance:
            best_layout = adjacent_layout
            best_distance = new_distance

        # Select worse layouts with some chance
        if new_distance < current_distance or random.random() < p:
            layout = adjacent_layout
            current_distance = new_distance

        # Reduce the randomness (temperature is proportional to it)
        temperature *= cooling_rate
        distances.append(current_distance)

    return best_layout, distances


def get_neighbour(layout: Layout) -> Layout:
    """
    Generates a layout that is "close" to the given input layout.
    Basically swaps the positions of two randomly chosen keys :)

    Parameters
    ----------
    layout: Layout
        The reference layout.

    Returns
    -------
    adjacent_layout: Layout
        A layout that is similar to the input (it is "slightly" different)
    """
    # NOTE: This whole mapping business is done here in order to avoid having to deep copy everything :)

    # Get 2 random keys to swap
    k1, k2 = random.sample(range(len(layout.mapping)), 2)

    # Remap said keys
    new_mapping = layout.mapping.copy()
    new_mapping[k1], new_mapping[k2] = new_mapping[k2], new_mapping[k1]

    # Create a new layout with the newly created mapping
    return layout.with_mapping(new_mapping)


def calculate_distance(layout: Layout, text: str) -> float:
    """
    Spawns a KeyboardAnalyzer and calculates the distance travelled
    (Exists just for convenience)

    Parameters
    ----------
    layout: Layout
        The layout to be used.

    text: str
        The text to measure distance travelled on

    Returns
    -------
    distance_travelled: float
        The distance travelled by the fingers for typing the text
    """
    kbda = KeyboardAnalyzer(layout)
    kbda.analyze(text)
    return kbda.get_distance_travelled()
