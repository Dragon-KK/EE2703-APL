import os
import sys
import warnings
import matplotlib.pyplot as plt

try:
    from ee23b135_simulated_annealing import optimize
    from ee23b135_kbda import KeyboardAnalyzer
    from ee23b135_layout import Layout

except ImportError as import_error:
    # Just in case the person using this didn't read the readme
    warnings.warn(
        "Did you forget to place `ee23b135_kbda.py`, `ee23b135_simulated_annealing` and `ee23b135_layout` in the current directory?",
        stacklevel=2,
    )
    raise import_error

SHOW_IMAGE = False
"""Set to true to show the heatmaps on generation (using `PIL.Image.Image.show`)"""

SAVE_IMAGE = True
"""Set to true to save the heatmaps generated"""

IMAGE_OUTPUT_DIRECTORY = "ee23b135_heatmaps"
"""The directory where heatmaps will be saved to (If `SAVE_IMAGE` is set to True)"""

IMAGE_OUTPUT_PREFIX = "ee23b135_heatmap"
"""The name with which heatmaps will be saved with (If `SAVE_IMAGE` is set to True)"""


def generate_outfile_name():
    """
    Generates file names of the type
    `{IMAGE_OUTPUT_DIRECTORY}/{IMAGE_OUTPUT_PREFIX}{NUMBER}.png`
    Where NUMEBER is the number of files present in `IMAGE_OUTPUT_DIRECTORY`

    (This is done in an effort to create unique file names that follow some order)

    Returns
    -------
    file_path: str
        A (hopefully unique) file path to save the image as :)
    """
    # Make the directory if it doesn't exist
    os.makedirs(IMAGE_OUTPUT_DIRECTORY, exist_ok=True)

    # Find the number of items in the directory
    number = len(os.listdir(IMAGE_OUTPUT_DIRECTORY))

    return f"{IMAGE_OUTPUT_DIRECTORY}/{IMAGE_OUTPUT_PREFIX}{number}.png"


if __name__ == "__main__":
    if len(sys.argv) > 1:
        layout_file_path = sys.argv[1]
    else:
        print("NOTE: You can provide the file path directly as a command line argument")
        layout_file_path = input("Enter a json file with the keyboard layout: ")

    # Load the initial layout
    layout = Layout.from_file(layout_file_path)

    if len(sys.argv) > 2:
        text_file_path = sys.argv[2]
    else:
        print("NOTE: You can provide the file path directly as a command line argument")
        text_file_path = input("Enter a file with the text data: ")

    try:  # Try opening the file and raise an error if it doesn't exist
        with open(text_file_path, "r") as f:
            text = f.read()

    except Exception:  # i.e. The file couldn't be opened for reading
        raise FileNotFoundError(f"The file `{text_file_path}` cannot be read!")

    print("Optimizing layout :)")

    # Find the optimized layout for the text
    # The optimize function takes more parameters, feel free to play around with them
    optimized_layout, distances = optimize(layout, text)

    # Create analyzer to generate plot if required
    kbda = KeyboardAnalyzer(optimized_layout)
    kbda.analyze(text, ignore_space=False, ignore_shift=False)

    heatmap = None
    if SHOW_IMAGE or SAVE_IMAGE:
        heatmap = kbda.generate_heatmap()

    print("Distance travelled in original layout:", distances[0])
    print("Distance travelled in optimized layout:", kbda.get_distance_travelled())

    if SAVE_IMAGE:
        assert heatmap is not None
        outfile = generate_outfile_name()
        print(f"Saving heatmap of optimized to `{outfile}`")
        heatmap.save(outfile)

    if SHOW_IMAGE:
        assert heatmap is not None
        print("Showing heatmap of optimized layout...")
        heatmap.show()

    # Plot the best distances reached at each step
    plt.plot(list(min(distances[: i + 1]) for i in range(len(distances))))  # type: ignore
    # Plot the intermediate steps
    plt.plot(distances)  # type: ignore

    plt.title("Simulated annealing top find optimal keyboard")  # type: ignore
    plt.xlabel("Iteration")  # type: ignore
    plt.ylabel("Distance")  # type: ignore

    plt.legend(["Best distance", "Current distance"])  # type: ignore
    plt.show()  # type: ignore
