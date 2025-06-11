import os
import sys
import warnings

try:
    from ee23b135_kbda import KeyboardAnalyzer
    from ee23b135_layout import Layout

except ImportError as import_error:
    # Just in case the person using this didn't read the readme
    warnings.warn("Did you forget to place `ee23b135_kbda.py` and `ee23b135_layout` in the current directory?", stacklevel=2)
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

if __name__ == '__main__':    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        print("NOTE: You can provide the file path directly as a command line argument")
        file_path = input("Enter a json file with the keyboard layout: ")

    # Load the layout and create the analyzer
    kbda = KeyboardAnalyzer(Layout.from_file(file_path))
    
    # Keep taking text input to analyze (Stop when the user gives an empty string)
    while True:
        text = input("Enter text to analyze (Give empty string to exit): ")
        if not text:
            break

        # Analyze the text and generate the heatmap
        kbda.analyze(text)
        heatmap = kbda.generate_heatmap()
    
        print("Distance travelled:", kbda.get_distance_travelled())

        if SAVE_IMAGE:
            outfile = generate_outfile_name()
            print(f"Saving heatmap to `{outfile}`")
            heatmap.save(outfile)

        if SHOW_IMAGE:
            print("Showing heatmap...")
            heatmap.show()

        # Analyzing the text changes the state of the kbda,
        # Reset it in order to analyze new set of text
        kbda.reset()
