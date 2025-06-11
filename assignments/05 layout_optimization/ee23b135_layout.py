"""
NOTE: This file is pretty much the same as the assignment 4 submission
The only new addition the usage of `mapped` lists (the definitions are made below)
"""

from typing import Any, Generic, Optional, TypeVar
import warnings
import json

try:
    import ee23b135_parsing_tools as ptools

except ImportError as import_error:
    # Just in case the person using this didn't read the readme
    warnings.warn(
        "Did you forget to place `ee23b135_parsing_tools.py` in the current directory?",
        stacklevel=2,
    )
    raise import_error


Key = int
"""Every key is represented by an integer index"""

Composition = tuple[Key, ...]
"""A group of keys that are to be pressed to compose a character"""

Position = tuple[float, float]
"""The coordinates of a key"""

DEDUCE_EXTRA_DISPLAYED_LETTERS = True
"""
Set to true in order to try and render the keys that have more than one letter shown on them properly
NOTE: Set this to False if the keyboard rendered seems to have too much additional text
"""

T = TypeVar("T")


class InputMappedList(Generic[T]):
    """
    A data stucture that maps the inputs to another set of input
    (Given by the permutation `mapping`)

    NOTE: This exists just to allow me to create adjacent layouts without having to copy all the data
    """

    def __init__(self, data: list[T], mapping: list[int]):
        self.mapping = mapping
        self.data = data

    def __getitem__(self, idx: int):
        return self.data[self.mapping[idx]]

    def __iter__(self):
        return (self.data[self.mapping[i]] for i in range(len(self.data))).__iter__()

    def copy(self):
        return InputMappedList(self.data.copy(), self.mapping.copy())


class OutputMappedList:
    """
    A data stucture that maps the outputs to another set of output
    (Given by the map `mapping`)

    NOTE: This exists just to allow me to create adjacent layouts without having to copy all the data
    """

    def __init__(self, data: list[int], mapping: list[int]):
        self.mapping = mapping
        self.data = data

    def __getitem__(self, idx: int):
        return self.mapping[self.data[idx]]

    def __iter__(self):
        return (self.mapping[self.data[i]] for i in range(len(self.data))).__iter__()

    def copy(self):
        return OutputMappedList(self.data.copy(), self.mapping.copy())


class Layout:
    def __init__(
        self,
        character_map: dict[str, Composition],
        position: list[Position],
        group: list[int],
        group_home: list[Key],
        key_text: list[tuple[str, ...]],
        mapping: Optional[list[int]] = None,
    ) -> None:
        msg = "Invalid layout definition, use factory functions to create Layout!"
        assert len(position) == len(group) == len(key_text), msg
        assert len(group_home) == max(group) + 1, msg
        assert len(position) != 0, msg

        if mapping is None:
            mapping = list(range(len(position)))

        self.mapping = mapping

        self.character_map = character_map
        """A mapping that tells what key presses a character is composed of"""

        # To get clarity on why some objects are mapped while others are not,
        # refer to the `analyze` function in kbda
        self.position = InputMappedList(position, mapping)
        """A mapping that tells the position of a key"""

        self.group = InputMappedList(group, mapping)
        """
        A mapping that tells which `finger group` a key belongs to.
        A finger group is a group of keys that are to be typed using the same finger
        """

        self.group_home = OutputMappedList(group_home, _invert(mapping))
        """
        A mapping that tells the `home key` for a certain `finger group`
        """

        self.key_text = key_text
        """
        A mapping that tells what all letters are shown on a key.
        NOTE: This is only used while rendering the keyboard :)
        """

    @classmethod
    def from_dict(cls, keys: dict[str, Any], characters: dict[str, Any]):
        """
        Generate layout object from the keys and characters dicts (specified in updated assignment)

        Parameters
        ----------
        keys: dict[str, KeysDictItem]
            A dictionary that describes the position and home key for every key

        characters: dict[str, list[str]]
            A dictionary that descibes what all keys need to be pressed for a character

        Raises
        ------
        InvalidLayoutDescriptionError
            - If the keys dict or the characters dict is invalid
            - If the layout provided is empty
        """

        # First we type check the data to ensure it follows the correct format
        validated_keys = ptools.validate_keys_dict(keys)
        validated_characters = ptools.validate_characters_dict(
            characters, validated_keys
        )

        # We the people condemn empty layouts are bad :)
        if len(validated_keys) == 0:
            raise ptools.InvalidLayoutDescriptionError("Layout is empty!")

        # Assigned each key an index and get the reverse mapping also (for convenience)
        key_to_idx_map = dict((key, idx) for idx, key in enumerate(validated_keys))
        idx_to_key_map = dict((idx, key) for key, idx in key_to_idx_map.items())

        # Populate the character map
        character_map: dict[str, tuple[Key, ...]] = {
            char: tuple(  # Sequence of key indices
                key_to_idx_map[key] for key in validated_characters[char]
            )
            for char in validated_characters
        }

        # Populate the positions map
        position = [
            validated_keys[idx_to_key_map[i]]["pos"] for i in range(len(validated_keys))
        ]

        # Setup to populate the groups map in a convenient manner
        group_idx = 0

        # A dictionary that provides an identifier associated with a `finger group`
        # Refer to `Layout.group` for more info
        group_dict: dict[str, int] = dict()
        group_home: list[Key] = []  # The starting position of fingers in the group

        # Populate the groups map
        group = [-1] * len(position)
        for key in range(len(group)):
            start_key = validated_keys[idx_to_key_map[key]]["start"]
            if group_dict.get(start_key) is None:
                group_dict[start_key] = group_idx
                group_home.append(key_to_idx_map[start_key])
                group_idx += 1

            group[key] = group_dict[start_key]

        # Populate the key_text with just the base key character
        # Basically a list that tells what all letters are displayed on the image of a key
        key_text: list[tuple[str, ...]] = [
            (idx_to_key_map[i],) for i in range(len(validated_keys))
        ]

        if DEDUCE_EXTRA_DISPLAYED_LETTERS:
            for char in validated_characters:
                # Only the non alphabetic letters (That aren't keys)
                # seem to have extra letters displayed on the key

                if char.lower() in validated_keys:
                    # All normal alphabetic letters and keys
                    # would have their `lower` present in keys already
                    continue

                for component in validated_characters[char]:
                    # Now I only want to show the additional letters on non special keys
                    # Notice that all special keys have a name of size > 1
                    if len(component) == 1:
                        # i.e component is a non special key that composes char
                        # (For example `5` composes `%')
                        key_text[key_to_idx_map[component]] += (char,)
                        break

        return cls(
            character_map=character_map,
            position=position,
            group=group,
            group_home=group_home,
            key_text=key_text,
        )

    @classmethod
    def from_file(cls, file_path: str):
        """
        Generate layout object from a json file

        Parameters
        ----------
        file_path: str
            Path to the json file with the layout description

        Raises
        ------
        FileNotFoundError
            - If the file given wasn't able to be read from

        JSONDecodeError
            - If the json file couldn't be parsed

        InvalidLayoutDescriptionError
            - If the layout is invalid
        """
        try:  # Try opening the file and raise an error if it doesn't exist
            with open(file_path, "r") as f:
                file_data = f.read()

        except Exception:  # i.e. The file couldn't be opened for reading
            raise FileNotFoundError(f"The file `{file_path}` cannot be read!")

        try:  # Try parsing the file and raise an error if it couldn't
            raw_layout = json.loads(file_data)

        except Exception as json_error:  # i.e. The file couldnt be parsed by json
            warnings.warn(
                f"The file `{file_path}` is not a valid json file!", stacklevel=2
            )
            raise json_error

        # Do some preliminary type checking
        # (before we can safely hand it over to `cls.from_dict`)
        raw_layout = ptools.validate_raw_layout(raw_layout)

        # Now that we have the keys and chracters dict create the Layout object
        return cls.from_dict(
            keys=raw_layout["keys"], characters=raw_layout["characters"]
        )

    def with_mapping(self, mapping: list[int]) -> "Layout":
        """
        Creates a new layout with a new mapping.
        NOTE: The underlying data (like the character map and position map etc. are not copied)

        Parameters
        ----------
        mapping: list[int]
            The new mapping to use

        Returns
        -------
        mapped_layout: Layout
            The copy of the given layout with the new mapping
        """
        return Layout(
            character_map=self.character_map,
            group_home=self.group_home.data,
            position=self.position.data,
            key_text=self.key_text,
            group=self.group.data,
            mapping=mapping,
        )


def _invert(permutation: list[int]):
    """Inverts a permutation"""
    inverted = [0] * len(permutation)
    for i in range(len(permutation)):
        inverted[permutation[i]] = i
    return inverted
