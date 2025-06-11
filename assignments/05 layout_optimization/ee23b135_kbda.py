"""
NOTE: This file is pretty much the same as the assignment 4 submission
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Callable

from PIL import Image, ImageDraw, ImageFilter, ImageFont

import matplotlib.font_manager

import numpy as np

if TYPE_CHECKING:
    from ee23b135_layout import Layout, Key, Position, InputMappedList, OutputMappedList

# Finds a font file to use in the system (Used while generating the keyboard)
font: str = matplotlib.font_manager.FontManager().defaultFont["ttf"]  # type: ignore

COLOR_SCHEME = {
    "dark": {"base": "#3d3d3d", "key": "#494949", "text": "#b2b2b2"},
    "light": {"base": "#d3d3d3", "key": "#e3e3e3", "text": "#3e3e3e"},
}["dark"]
"""
The colors used in drawing the keyboard
(Feel free to use any)
"""

KEYBOARD_RESOLUTION = 750
"""
Feel free to change resolution in order to increase or decrease image quality
(You might have to change the aspect ration of KEYBOARD_SIZE if the keyboard looks wonky)
"""

HEATMAP_CONTOURING = 21
"""
A value between 1 and 255 (inclusive)
The smaller the number, the more contours there are (Which then makes them less visible)
*NOTE*: Some nice values to use are 8, 21, 32
"""

IGNORE_SHIFT = False
"""Set to true to ignore the shift key while making the heatmap"""

IGNORE_SPACE = True
"""Set to true to ignore the space key while making the heatmap"""

HEATMAP_QUALITY = 0.6
"""
A scaling factor that effects the heatmap generation resolution
*NOTE*: Basically a scaled down version of the heatmap is generated then that is scaled up 
and merged with the keyboard image.
*NOTE*: The bigger this number the more time it will take to generate a heatmap.
"""

HEATMAP_FREQUENCY_CUTOFF = 0.1
"""
If the frequence of a character is less that `HEATMAP_FREQUENCY_CUTOFF` of the mean,
it is skipped in the heatmap
"""

KEYBOARD_SIZE = (3 * KEYBOARD_RESOLUTION, KEYBOARD_RESOLUTION)
"""
The image dimensions of the heatmap generated
I noticed that a 3/1 ratio looked good while displaying the qwerty keyboard
Play around with this constant and `KEY_PREFERRED_HEIGHT` to get a nice looking keyboard
"""

KEY_PREFERRED_HEIGHT = KEYBOARD_SIZE[1] / 6
"""
The ideal height of a key (int px)
Play around with this parameter if the keyboard looks wonky
"""

NORMAL_KEY_PREFERRED_WIDTH = KEY_PREFERRED_HEIGHT
"""The amount of space a normal key would ideally like to fill :)"""

SHIFT_KEY_PREFERRED_WIDTH = 2.5 * NORMAL_KEY_PREFERRED_WIDTH
"""The amount of space the shift key would ideally like to fill :)"""

SPACEBAR_PREFERRED_WIDTH = 5 * NORMAL_KEY_PREFERRED_WIDTH
"""The amount of space the spacebar would ideally like to fill :)"""

KEYBOARD_BBOX_PADDING = KEYBOARD_SIZE[1] / 20
"""The extra padding (in px) given to the keyboard image (on the boundaries)"""

KEY_PADDING = KEY_PREFERRED_HEIGHT / 9
"""The space between keys"""

MINIMUM_KEY_SIZE = 10.0
"""
The minimum size a key can be
Ideally this shoudl never be used (Just give a better layout bro)
"""


class KeyboardAnalyzer:
    def __init__(self, layout: Layout):
        self._layout = layout

        self._finger_locations: OutputMappedList = self._layout.group_home.copy()
        self._key_frequency = [0] * len(self._layout.mapping)
        self._distance_travelled = 0.0

    def reset(self):
        """
        Resets the state of the KeyboardAnalyzer to its inital condition.
        *NOTE*: Use this before if you analyze new text
        """
        self._finger_locations = self._layout.group_home.copy()
        self._key_frequency = [0] * len(self._layout.mapping)
        self._distance_travelled = 0.0

    def analyze(
        self,
        text: str,
        ignore_space: bool = IGNORE_SPACE,
        ignore_shift: bool = IGNORE_SHIFT,
    ):
        """
        Analyzes the keyboard using the text.
        *NOTE*: This changes the state of the analyzer
        (Use `reset` before analyzing more text)

        Parameters
        ----------
        text : str
            The text used to analyze the keyboard.

            Every character in text must have been defined in the layout!

        Raises
        ------
        ValueError
            - If the text contains an unrecognized character
        """
        # Fill in the frequency array and also calculate the finger distance travelled
        for char in text:
            if self._layout.character_map.get(char) is None:
                # We could also decide to just ignore unrecognized characters
                raise ValueError(f"ERROR! Unrecognized character `{char}`")

            for key in self._layout.character_map[char]:
                group = self._layout.group[
                    key
                ]  # The finger responsible to type the letter
                start_key = self._finger_locations[
                    group
                ]  # The current position of said finger

                # Realize that here we would actually want the new key that is at the finger location
                # That is why layout.group_home is output mapped (with the inversion of `mapping`)

                # It is obvious here that we need to position of the keys in the new mapping :)
                start_pos = self._layout.position[start_key]
                final_pos = self._layout.position[key]

                self._distance_travelled += _euclid_distance(start_pos, final_pos)

                # In the model described in the quiz, the finger goes from its home to the key
                # then proceeds to teleport back to the home row (with 0 distance travelled)

                if ignore_space and any(
                    v.lower().startswith("space") for v in self._layout.key_text[key]
                ):
                    # Do not increment frequency if the character was special
                    continue

                if ignore_shift and any(
                    v.lower().startswith("shift") for v in self._layout.key_text[key]
                ):
                    # Do not increment frequency if the character was special
                    continue

                # Fill in the frequency array based on the occurances of characters
                self._key_frequency[key] += 1

    def get_distance_travelled(self) -> float:
        """
        Gets the distance travelled by fingers till now
        *NOTE*: Calling this beofre calling `analyze` is probably not what you want to do

        Returns
        -------
        distance_travelled: float
            The distance travelled by the fingers for typing the text given to analyze
        """
        return self._distance_travelled

    def generate_heatmap(self) -> Image.Image:
        """
        Generates the heatmap using the frequency array and overlays it on the keyboard.

        NOTE: This may seem slow (due to it being run on python)
        One may take the same approach `heatmap.py` did and move this function to c

        Returns
        ------
        heatmap: PIL.Image
            The image of the heatmap generated based on the frequency array.
        """
        keyboard_image, key_centres = self._generate_keyboard()

        # Get the size of the heatmap (Usually scaled down as we don't need to use high pixel density to get good results)
        width, height = tuple(int(v * HEATMAP_QUALITY) for v in KEYBOARD_SIZE)

        total_frequency = sum(self._key_frequency)

        if total_frequency == 0:  # i.e No keys were pressed
            return keyboard_image

        # Just having a normal mean didn't seem like a good metric (It didn't look that useful)
        # Hence I calculate the mean frequency of all keys pressed
        mean_frequency = total_frequency / sum(freq > 0 for freq in self._key_frequency)
        max_frequency = max(self._key_frequency)

        # 1D array that has a lower value for more frequent locations (0-255)
        heatmap = np.full((height, width), 255, dtype=np.int32)  # type: ignore

        for (x, y), freq in (
            (key_centres[key], self._key_frequency[key])
            for key in range(len(self._key_frequency))
        ):
            if (freq == 0) or freq < mean_frequency * HEATMAP_FREQUENCY_CUTOFF:
                # If the key didn't appear frequent enough just ignore it
                continue

            # The amount of radius a key will influence
            # (The constants used were found by trial and error. I just kept tweaking the values till the image seemed nice)
            FUZZ = int(
                (
                    KEY_PREFERRED_HEIGHT * HEATMAP_QUALITY
                )  # The radius must be proportional to the key size (in heatmap space)
                * min(
                    total_frequency / 15, 1.5
                )  # I wanted the heatmap to be more tame for text with less text
                * min(
                    max(0.3, freq * 1.5 / mean_frequency), 1.1
                )  # A key will have a higher radius of influence the more frequent it was used in proportion to the others
            )

            # Map the coordinates to the reduced heatmap space
            x *= HEATMAP_QUALITY
            y *= HEATMAP_QUALITY

            # Iterate in all pixels in the circular region aroudn x, y (radius=FUZZ)
            for i in range(int(x) - FUZZ, int(x) + FUZZ):
                for j in range(int(y) - FUZZ, int(y) + FUZZ):
                    if j < 0 or i < 0 or j >= height or i >= width:
                        continue  # i.e The pixel is not in bounds of the image

                    dist = ((i - x) ** 2 + (j - y) ** 2) ** 0.5
                    if dist > FUZZ:
                        continue  # i.e The pixel is not in the circular region

                    # A ratio (smaller it is the higher value it will get on the heatmap)
                    # Again the formula and constants were formed with trial and error

                    # The closer the frequency is to max_frequency, the smaller addition is done to dist
                    # Smaller addition to dist implies that the ratio (pixVal) will be smaller (ie higher value on heatmap)
                    pixVal = (dist + FUZZ * min(max_frequency / freq - 1, 1) / 4) / FUZZ

                    # Some extra dependancy on the frequency (wrt total frequency this time)
                    pixVal *= max(total_frequency / (freq), 1) ** 0.2

                    # Decrement the current pixel value by some ratio
                    heatmap[j][i] *= min(pixVal, 1)

        # In order to now generate some contour lines
        # I make all numbers with the same quotient (when divided by HEATMAP_COUNTOURING) to the same value
        heatmap = (heatmap // HEATMAP_CONTOURING) * HEATMAP_CONTOURING + (
            255 % HEATMAP_CONTOURING
        )

        blurred_image = (  # Create an image (after inverting the above created heatmap)
            Image.fromarray(np.uint8(255 - heatmap))
            .resize(KEYBOARD_SIZE)  # Resize it to the size of the keyboard
            .filter(  # Apply a gaussian blur to make it smoother
                ImageFilter.GaussianBlur(
                    # The bigger the number, the less it smoothens
                    KEY_PREFERRED_HEIGHT
                    / min(100, 4 * HEATMAP_CONTOURING)
                )
            )
            .convert("RGBA")  # Add the extra channels (filled later)
        )
        blurred_image_data = np.array(blurred_image)  # type: ignore

        # NOTE: After some testing I realized that iterating all pixels in the image was very slow in python
        # Using numpy arrays fastened up the process by a low (ig because it is written in c?)
        # Below I give rgb values to pixels based on the greyscale values found above

        # Store the gradient information in the alpha channel for now
        blurred_image_data[:, :, 3] = blurred_image_data[:, :, 0]
        blurred_image_data[:, :, :3] = 0  # Clear the rgb channels

        # I make 3 groups of pixels: 0 -> LOW_CUTOFF, LOW_CUTOFF -> HIGH_CUTOFF, HIGH_CUTOFF -> 255
        HIGH_CUTOFF = 150
        LOW_CUTOFF = 80

        # Create masks (set to 1 if the pixel belongs the the group)
        high_mask = np.greater(blurred_image_data[:, :, 3], HIGH_CUTOFF)
        medium_mask = np.greater(blurred_image_data[:, :, 3], LOW_CUTOFF) * (
            high_mask == 0
        )
        low_mask = high_mask | medium_mask == 0

        # Normalize the values in each group to be between 0 to 1
        high_normalized = (blurred_image_data[:, :, 3] - HIGH_CUTOFF) / (
            255 - HIGH_CUTOFF
        )
        medium_normalized = (blurred_image_data[:, :, 3] - LOW_CUTOFF) / (
            HIGH_CUTOFF - LOW_CUTOFF
        )
        low_normalized = (blurred_image_data[:, :, 3] - 0) / (LOW_CUTOFF)

        # Now simply linearly interpolate the rgb values based on their normalized value
        # blurred_image_data[:, :, i] = np.minimum(255, blurred_image_data[:, :, i] + mask * (normalized * (high - low) + (low)))
        # This line basically adds `(normalized * (high - low) + (low))` # Linear interpolation moment
        # to blurred_image_data[:, :, i] only if the mask was 1
        # Then blurred_image_data[:, :, i] is clamped to be atmost 255

        # The numbers used are based on the colors I wanted :)

        # Higher values go fro orangeish to like reddish
        # orangeish:        #efef78
        # reddish:          #df4b66
        blurred_image_data[:, :, 0] = np.minimum(
            255,
            blurred_image_data[:, :, 3]
            + high_mask * (high_normalized * (0xDF - 0xEF) + (0xEF)),
        )
        blurred_image_data[:, :, 1] = np.minimum(
            255,
            blurred_image_data[:, :, 1]
            + high_mask * (high_normalized * (0x4B - 0xEF) + (0xEF)),
        )
        blurred_image_data[:, :, 2] = np.minimum(
            255,
            blurred_image_data[:, :, 2]
            + high_mask * (high_normalized * (0x66 - 0x78) + (0x78)),
        )

        # Medium values go from greenish to yellowish
        # green:            #4fc46c
        # yellowish:        #efef78
        blurred_image_data[:, :, 0] = np.minimum(
            255,
            blurred_image_data[:, :, 0]
            + medium_mask * (medium_normalized * (0xEF - 0x4F) + (0x4F)),
        )
        blurred_image_data[:, :, 1] = np.minimum(
            255,
            blurred_image_data[:, :, 1]
            + medium_mask * (medium_normalized * (0xEF - 0xC4) + (0xC4)),
        )
        blurred_image_data[:, :, 2] = np.minimum(
            255,
            blurred_image_data[:, :, 2]
            + medium_mask * (medium_normalized * (0x78 - 0x6C) + (0x6C)),
        )

        # # Low values go from violet to light blueish
        # violet:           #450c53
        # light blueish:    #5bb1ce
        blurred_image_data[:, :, 0] = np.minimum(
            255,
            blurred_image_data[:, :, 0]
            + low_mask * (low_normalized * (0x5B - 0x45) + (0x45)),
        )
        blurred_image_data[:, :, 1] = np.minimum(
            255,
            blurred_image_data[:, :, 1]
            + low_mask * (low_normalized * (0xB1 - 0x0C) + (0x0C)),
        )
        blurred_image_data[:, :, 2] = np.minimum(
            255,
            blurred_image_data[:, :, 2]
            + low_mask * (low_normalized * (0xCE - 0x53) + (0x53)),
        )

        # Set the alpha to 0 for anything that was blackish, and set it to 0.4 for the rest of the image
        blurred_image_data[:, :, 3] = 255 * 0.4 * (blurred_image_data[:, :, 3] > 10)

        # Finally composite the keyboard image with this heatmap generated :)
        return Image.alpha_composite(
            keyboard_image, Image.fromarray(blurred_image_data)
        )

    def _generate_keyboard(self) -> tuple[Image.Image, list[Position]]:
        """
        Generates the image of the keyboard based on the layout

        Returns
        ------
        keyboard_image: PIL.Image
            The image generated

        key_centres: list[Position]
            The centres of each key in image space (Used while rendering the heatmap)
        """
        base = Image.new("RGBA", KEYBOARD_SIZE)
        canvas = ImageDraw.Draw(base)

        # Draw a big rectangle that covers the full image (The keyboard base)
        canvas.rounded_rectangle(
            ((0, 0), KEYBOARD_SIZE),
            radius=KEYBOARD_BBOX_PADDING,
            fill=COLOR_SCHEME["base"],
        )
        # Get the row layout
        row_layout = _get_rowwise_layout(self._layout.position)
        rows = sorted(row_layout.keys())  # Row levels (ASC)

        # The font used while drwing the key text
        kb_font = ImageFont.FreeTypeFont(font, int(7 * KEY_PREFERRED_HEIGHT / 24))

        # The function used to convert coordinates from input space to image space
        _map_coordinates, xscale, yscale = _deduce_space_mapping(
            self._layout, row_layout
        )

        # The centre of each key in image space (Populated later)
        key_centres: list[Position] = [(0, 0)] * len(self._layout.position.data)

        for row_idx in range(len(rows)):
            if row_idx != 0:  # For every row except the first (the bottom most)
                # Fill till the next row leaving some padding
                # Or take preferred height if possible
                key_height = min(
                    (rows[row_idx] - rows[row_idx - 1]) * yscale - KEY_PADDING,
                    KEY_PREFERRED_HEIGHT,
                )

            else:  # The bottom most row can always take their preferred height
                key_height = KEY_PREFERRED_HEIGHT

            key_height = max(MINIMUM_KEY_SIZE, key_height)

            row = row_layout[rows[row_idx]]
            for col_idx in range(len(row)):
                if col_idx != len(row) - 1:  # For every col except the last (rightmost)
                    x1 = self._layout.position[row[col_idx]][0]
                    x2 = self._layout.position[row[col_idx + 1]][0]

                    # Fill till the next col leaving some padding
                    # Or take preferred width if possible
                    key_width = min(
                        (x2 - x1) * xscale - KEY_PADDING,
                        _get_preferred_width(row[col_idx], self._layout),
                    )

                else:  # The last column can always take their preferred widht
                    key_width = _get_preferred_width(row[col_idx], self._layout)

                key_width = max(MINIMUM_KEY_SIZE, key_width)

                key = row[col_idx]

                # The bbox of the key
                top_left = _map_coordinates(self._layout.position[key])
                bottom_right = (top_left[0] + key_width, top_left[1] + key_height)
                key_centres[key] = (
                    (top_left[0] + bottom_right[0]) / 2,
                    (top_left[1] + bottom_right[1]) / 2,
                )
                # Draw a rectangle for the key
                canvas.rounded_rectangle(
                    (top_left, bottom_right),
                    fill=COLOR_SCHEME["key"],
                    radius=key_height // 8,
                )
                # Draw all the text on the key
                canvas.text(  # type: ignore
                    key_centres[key],
                    "\n".join(self._layout.key_text[key][::-1]),
                    font=kb_font,
                    fill=COLOR_SCHEME["text"],
                    align="center",
                    spacing=key_height / 5,
                    anchor="mm",
                )

        return base, key_centres


def _get_preferred_width(key: Key, layout: Layout) -> float:
    """
    Finds the preferred width a key would like to occupy

    Parameters
    ----------
    key: Key
        The key in question

    layout: Layout
        The keyboard layout

    Returns
    ------
    width: float
        The preferred width
    """
    if any(label.lower().startswith("shift") for label in layout.key_text[key]):
        return SHIFT_KEY_PREFERRED_WIDTH

    if any(label.lower().startswith("space") for label in layout.key_text[key]):
        return SPACEBAR_PREFERRED_WIDTH

    return NORMAL_KEY_PREFERRED_WIDTH


def _deduce_space_mapping(
    layout: Layout, row_layout: dict[float, list[Key]]
) -> tuple[Callable[[Position], Position], float, float]:
    """
    Finds the function that maps the coordinates given as input to image coordinates.
    Takes into account the padding and resolution of the heatmap and the preferred size
    of keys

    Parameters
    ----------
    layout: Layout
        The keyboard layout

    row_layout: dict[float, list[Key]]
        The rowwise list of keys (Generated by `_get_rowwise_layout`)

    Returns
    ------
    mapping: Callable[[Position], Position]
        A function that maps coordinates from input space to the image space

    x_scaling_factor: float
        The factor used to scale width from input space to image space

    y_scaling_factor: float
        The factor used to scale height from input space to image space
    """
    # The minimum and maximum
    minx = min(layout.position[row[0]][0] for row in row_layout.values())
    maxx = max(layout.position[row[-1]][0] for row in row_layout.values())
    miny = min(row_layout.keys())
    maxy = max(row_layout.keys())

    # Get the approximate scaling factor (image space to input space) which is then
    # used to find the bbox of the keyboard
    approx_x_scaler = (maxx - minx + 1) / KEYBOARD_SIZE[0]
    approx_y_scaler = (maxy - miny + 1) / KEYBOARD_SIZE[1]
    # NOTE: The extra adding of 1 is done to handle cases where maxi = mini
    # (This is equivalent to assuming the last key has a size of 1 in input space)

    right_bound = max(  # Get the right bound in input coordinate space
        layout.position[row[-1]][0]
        + (_get_preferred_width(row[-1], layout) * approx_x_scaler)
        for row in row_layout.values()
    )

    bottom_bound = miny - KEY_PREFERRED_HEIGHT * approx_y_scaler

    # The keyboard bounds are now minx->right_bound and bottom_bound->maxy
    # this implies (right_bound - minx ) * x_scaling_factor + 2 * padding = image_width
    #              (maxy - bottom_bound) * y_scaling_factor + 2 * padding = image_height
    x_scaling_factor = (KEYBOARD_SIZE[0] - 2 * KEYBOARD_BBOX_PADDING) / (
        right_bound - minx
    )
    y_scaling_factor = (KEYBOARD_SIZE[1] - 2 * KEYBOARD_BBOX_PADDING) / (
        maxy - bottom_bound
    )

    def mapping(input_position: Position) -> Position:
        # Since in the image space the origin is on the top left, we invert the y
        return (
            (input_position[0] - minx) * x_scaling_factor + KEYBOARD_BBOX_PADDING,
            KEYBOARD_SIZE[1]
            - (
                (input_position[1] - bottom_bound) * y_scaling_factor
                + KEYBOARD_BBOX_PADDING
            ),
        )

    return mapping, x_scaling_factor, y_scaling_factor


def _get_rowwise_layout(positions: InputMappedList[Position]) -> dict[float, list[Key]]:
    """
    Generates a structure that stores keys based on their position.

    *NOTE*: The list of keys at a y level is sorted(ASC) by their x coordinate

    Parameters
    ----------
    positions: list[Position]
        The position of every key in the layout

    Returns
    ------
    row_layout: dict[float, list[Key]]
        A dict mapping from y level to list of Keys at that level (sorted by their x)
    """
    # Sort based off of y coordinate first then x coordinate (ASC)
    sorted_keys: list[tuple[Key, Position]] = sorted(
        enumerate(positions), key=lambda xs: (xs[1][1], xs[1][0])
    )
    keyboard_layout: dict[float, list[Key]] = {}

    # Populate the rowwise layout now that we have a nice sorted list of keys
    current_ylevel: float | None = None
    for key, position in sorted_keys:
        if position[1] == current_ylevel:
            # We are still at the same level as the previous key
            keyboard_layout[position[1]].append(key)

        else:
            # We reached a new level, create the list and add it
            current_ylevel = position[1]
            keyboard_layout[position[1]] = [key]

    return keyboard_layout


def _euclid_distance(pos1: Position, pos2: Position) -> float:
    """
    Finds the distance between two points.

    Returns
    ------
    distance: float
        The distance between the two points :)
    """
    # assert len(pos1) == len(pos2)
    return sum((pos1[i] - pos2[i]) ** 2 for i in range(len(pos1))) ** 0.5
