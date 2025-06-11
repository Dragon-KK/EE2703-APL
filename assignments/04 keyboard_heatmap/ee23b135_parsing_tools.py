from typing import Any, Callable, TypeVar, TypedDict, cast

T = TypeVar("T")

class InvalidLayoutDescriptionError(ValueError):
    """Exception raised when the layout description provided was erroneous"""
    def __init__(self, msg: str) -> None:
        super().__init__("The layout description provided is not valid! " + msg)


class RawLayoutDescription(TypedDict):
    """The expected type of the data json reads from the input file"""
    keys: dict[str, Any]
    characters: dict[str, Any]


class KeysDictItem(TypedDict):
    """The expected type of every value in the `keys` dictionary"""
    pos: tuple[float, float]
    start: str


def expect_key(key: str, obj: dict[Any, Any], *vtype: type, err_msg: str = ""):
    """
    Checks whether the object provided follows the following structure
    ```
    {
        key: AnyOf(vtype),
        ...
    }
    ```

    Raises
    ------
        InvalidLayoutDescriptionError
            - If obj[key] is not of vtype
    """
    if obj.get(key) is None or not isinstance(obj[key], vtype):
        raise InvalidLayoutDescriptionError(err_msg)

def try_parse(item: Any, parser: Callable[[Any], T], err_msg: str = "") -> T:
    """
    Returns parser(item)

    Raises
    ------
        InvalidLayoutDescriptionError
            - If `parser` throws an exception
    """
    try:
        return parser(item)
    except Exception:
        raise InvalidLayoutDescriptionError(err_msg)
    
def validate_raw_layout(raw_layout: Any):
    """
    Checks whether the object provided follows the following structure
    ```
    {
        "keys": dict,
        "characters": dict,
        ...
    }
    ```

    Raises
    ------
        InvalidLayoutDescriptionError
            - If the object doesn't follow the structure
    """
    # Ensure that it is a dictionary
    if not isinstance(raw_layout, dict):
        err = "Expected top level object to be a dictionary!"
        raise InvalidLayoutDescriptionError(err)

    # Ensure that it contains the following keys
    for key in ("keys", "characters"):
        err = f"Expected dictionary `{key}` in top level object!"
        expect_key(key, cast(dict[str, Any], raw_layout), dict, err_msg=err)

    return cast(RawLayoutDescription, raw_layout)

def validate_keys_dict(keys: dict[str, Any]):
    """
    Validates the data in the keys dict.
    Checks whether the object provided follows the following structure
    ```
    {
        *key<string>: {
            "pos": tuple[float, float],
            "start": keyof(keys),
            ...
        }
    }
    ```
    
    Raises
    ------
        InvalidLayoutDescriptionError
            - If the structure of the object is not as expected
            - If the home key for any key is not a key described
    """

    for key, value in keys.items():
        if not isinstance(value, dict):
            err = "Expected description of each key to be a dictionary!"
            raise InvalidLayoutDescriptionError(err)
        
        value = cast(dict[str, Any], value) # Anything for pylance to shut up

        # Ensure that the pos is of the correct type
        err = f"Expected keys[`{key}`][\"pos\"] to be a pair(tuple | list) of floats!"
        expect_key("pos", value, list, tuple, err_msg=err)

        # Ensure that pos is of size two (x and y coordinate)
        if len(value["pos"]) != 2:
            raise InvalidLayoutDescriptionError(err)
        
        # Also convert it to a tuple[float, float] (just to be consistent)
        value["pos"] = try_parse(
            value["pos"], lambda x: (float(x[0]), float(x[1])), err_msg=err
        )

        # Ensure that start is a valid key
        err = f"Expected keys[`{key}`][\"start\"] to be a key!"
        if value.get("start") is None or keys.get(value["start"]) is None:
            raise InvalidLayoutDescriptionError(err)
        
    return cast(dict[str, KeysDictItem], keys)

def validate_characters_dict(characters: dict[str, Any], keys: dict[str, KeysDictItem]):
    """
    Validates the data in the characters dict.
    Checks whether the object provided follows the following structure
    ```
    {
        *key<char>: tuple[keyof(keys), ...]
    }
    ```
    
    Raises
    ------
        InvalidLayoutDescriptionError
            - If the structure of the object is not as expected
            - If the composition of any character contains unrecognized keys
    """
    for char, value in list(characters.items()):
        err = f"Expected single character but found `{char}` in characters dict!"
        # Ensure that the character is indeed a single character
        if len(char) != 1:
            raise InvalidLayoutDescriptionError(err)
        
        # Ensure that the value is a sequence
        err = f"Expected composition of characters[`{char}`] to be a sequence(tuple | list) of keys!"
        if not isinstance(value, (list, tuple)):
            raise InvalidLayoutDescriptionError(err)
        
        value = cast(list[Any], value)

        # Ensure that each element of value is a valid key
        for key in value:
            if keys.get(key) is None:
                err += f" But found `{key}` (key doesn't exist) as composite element."
                raise InvalidLayoutDescriptionError(err)
        
        # Convert it into a tuple just to be consistent :)
        characters[char] = tuple(value)

    return cast(dict[str, tuple[str, ...]], characters)
