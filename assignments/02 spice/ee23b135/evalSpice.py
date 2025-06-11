"""
Solves the problem statement defined in Assignment 2 (SPICE Simulation).

Author:
    Kaushik G Iyer (EE23B135)
"""

from enum import Enum, auto
import sys
from typing import Callable, List, Optional, Tuple, TypeVar

import numpy as np


class Element(Enum):
    VOLTAGE_SOURCE = auto()
    CURRENT_SOURCE = auto()
    RESISTOR = auto()


SHOW_WARNINGS = True
"""Set to false to disable all warning messages"""

ValueType = np.float64
# float64 is more convenient to work with as it shows warnings when we divide by 0

PartialNode = str
PartialEdge = Tuple[PartialNode, PartialNode]
PartialBranch = Tuple[PartialEdge, Element, ValueType]

Node = int
Edge = Tuple[Node, Node]
Branch = Tuple[Edge, Element, ValueType]


def evalSpice(filename: str) -> Tuple[dict[str, float], dict[str, float]]:
    """
    Evaluates the unknowns in a circuit defined by the `SPICE` format.

    Parameters
    ----------
    filename: str
        Path to the file containing the circuit definition

    Returns
    -------
    node_voltages: dict[str, np.float64]
        A dictionary that tells the voltage at a specific node

        NOTE: The values are relative to GND (i.e. `V(GND) = 0`)

    source_currents: dict[str, np.float64]
        A dictionary that tells the current flowing through a voltage source branch

        NOTE: For a voltage source defined by `Vi nodeA nodeB ...`
        the current is +X if +X amps flows from nodeA to nodeB

    Raises
    ------
    FileNotFoundError
        - If the filename provided was not the path to a SPICE file
    ValueError
        - If the SPICE file provided couldn't be parsed
        - If the circuit is unsolvable
    """
    try:
        with open(filename, "r") as f:
            file_data = f.read()
    except Exception:
        raise FileNotFoundError("Please give the name of a valid SPICE file as input")

    ckt = Circuit.from_ckt_file(file_data)
    return ckt.solve()


class Circuit:
    def __init__(
        self, nodes: List[str], branches: dict[str, Branch], ground_index: Node
    ):
        self.nodes = nodes
        self.branches = branches
        self.ground_index = ground_index

    def solve(self) -> Tuple[dict[str, float], dict[str, float]]:
        """
        Evaluates the unknowns in the circuit

        Returns
        -------
        node_voltages: dict[str, np.float64]
            A dictionary that tells the voltage at a specific node

            NOTE: The values are relative to GND (i.e. `V(GND) = 0`)

        source_currents: dict[str, np.float64]
            A dictionary that tells the current flowing through a voltage source branch

            NOTE: For a voltage source defined by `Vi nodeA nodeB ...`
            the current is +X if +X amps flows from nodeA to nodeB

        Raises
        ------
        ValueError
            - If no ground node was given for reference
            - If the circuit is unsolvable
        """
        if self.ground_index == -1:
            if len(self.nodes) == 0:
                warn("WARNING: Circuit is empty!")
                return {}, {}

            warn("WARNING: No GND node provided for reference!")
            if any(node.upper() == "GND" for node in self.nodes):
                warn(
                    "NOTE: This implementation is case sensitive! Use `GND` to define ground node"
                )

            raise ValueError("Circuit error: no solution")

        # A mapping from source name to an index (Used while forming the `coeff_matrix`)
        voltage_source_map = dict(
            (name, i + len(self.nodes))
            for i, name in enumerate(
                name
                for name, branch in self.branches.items()
                if branch[1] == Element.VOLTAGE_SOURCE
            )
        )

        # The number of unknowns to sovle for
        unknowns = len(self.nodes) + len(voltage_source_map)

        # Corresponds to G, for Gx = y
        coeff_matrix = np.array(
            [[0] * unknowns for _ in range(unknowns)], dtype=ValueType
        )

        # Corresponds to y, for Gx = y
        ordinate_vector = np.array([0] * unknowns, dtype=ValueType)

        # refer to https://cheever.domains.swarthmore.edu/Ref/mna/MNA3.html
        # for information on why this construction is valid
        # The first len(nodes) unknowns are the node voltages
        # The remaining unknowns are the source currents

        # Populate the matrix
        for name, branch in self.branches.items():
            edge, element, value = branch
            node1, node2 = edge

            if element == Element.RESISTOR:
                assert value != 0
                # Recall that 0 resistance resistors are treated as 0 volt sources

                coeff_matrix[node1][node1] += 1 / value
                coeff_matrix[node2][node2] += 1 / value
                coeff_matrix[node1][node2] -= 1 / value
                coeff_matrix[node2][node1] -= 1 / value

            elif element == Element.VOLTAGE_SOURCE:
                # NOTE: Source defined by `Vx node1 node2 ...`
                # Implies that V(node1) - V(node2) = Vx
                coeff_matrix[node1][voltage_source_map[name]] = 1
                coeff_matrix[node2][voltage_source_map[name]] = -1
                coeff_matrix[voltage_source_map[name]][node1] = 1
                coeff_matrix[voltage_source_map[name]][node2] = -1

                ordinate_vector[voltage_source_map[name]] = value

            elif element == Element.CURRENT_SOURCE:
                # NOTE: Source defined by `Ix node1 node2 ...`
                # Implies that current from node1 to node2 = Ix
                ordinate_vector[node1] += -value
                ordinate_vector[node2] += value

            else:
                raise NotImplementedError(f"Unhandled element `{element}`")

        # The GND eqn must be V(GND) = 0
        for i in range(unknowns):
            coeff_matrix[self.ground_index][i] = 0
            coeff_matrix[i][self.ground_index] = 0
        coeff_matrix[self.ground_index][self.ground_index] = 1
        ordinate_vector[self.ground_index] = 0

        try:
            solution = np.linalg.solve(coeff_matrix, ordinate_vector)
        except np.linalg.LinAlgError:
            raise ValueError("Circuit error: no solution")

        node_voltages: dict[str, float] = {}
        source_currents: dict[str, float] = {}

        # Populate the node voltages that have been solved
        for i, value in enumerate(solution[: len(self.nodes)]):
            node_voltages[self.nodes[i]] = float(value)

        # Populate the source currents that have been solved
        for src, i in voltage_source_map.items():
            if src[0] != "V":
                # Since we handle 0 resistance resistors by treating them as
                # 0 volt sources, we must remember to discount those as sources
                # while creating the result dictionary (even though this is useful info)
                # (The problem statement only asked for current through voltage sources)
                continue

            source_currents[src] = float(solution[i])

        return node_voltages, source_currents

    @classmethod
    def from_branch_data(cls, data: dict[str, PartialBranch]):
        """
        Creates a Circuit from partial branch data.
        Also handles 0 resistance resistors in a special way.

        We basically assign each node an index and then update the branches to use
        these indices.
        This is done since it is more convenient to refer to a node by its index rather
        than its name

        Parameters
        ----------
        data: dict[str, PartialBranch]
            A dictionary describing every branch in the circuit

        Returns
        -------
        circuit: Circuit
            The newly created circuit object
        """
        node_map: dict[str, Node] = {}  # A mapping from name to index
        nodes: List[str] = []  # The list of all nodes
        branches: dict[str, Branch] = {}

        # A set containing all the edge on which 0 resistance wires exist
        wires: set[Edge] = set()

        for key, value in data.items():
            for i in range(2):  # Go through the two nodes in the edge
                # If the node wasn't seen before
                if value[0][i] not in node_map:
                    # Assign the node an index
                    node_map[value[0][i]] = len(nodes)
                    nodes.append(value[0][i])

            # 0 resistance resistors are treated as 0 volt sources
            # (to avoid division by 0 while creating the coeff matrix)
            if value[1] == Element.RESISTOR and value[2] == 0:
                edge = (node_map[value[0][0]], node_map[value[0][1]])

                # If there was already a 0 resistance wire in the edge,
                # There is no need to add another
                if edge in wires:
                    warn(
                        "WARNING: Multiple wires between same nodes! (You technically can't solve for current through each wire here.)"
                    )
                    continue

                # Below is the reason for which this check must be done

                # Let us consider a case where we allow for multiple wires bw nodes
                # Since we treat wires as 0 volt source,
                # If we have 2 wires from n1 to n2,
                # We would treat it as 2 `0 volt sources` from n1 to n2

                # But notice that here we can't solve for current in each source
                # Even in the circuit we can't solve for the current in each wire,

                # But... we can still solve for voltages
                # (which is what was asked)

                # By not placing multiple wire branches we bypass this problem :)

                branches[key] = (
                    edge,
                    Element.VOLTAGE_SOURCE,
                    ValueType(0),
                )

                # Update the set of wires
                wires.add(edge)
                wires.add((edge[1], edge[0]))
                continue

            # Refer to the node by its index
            branches[key] = ((node_map[value[0][0]], node_map[value[0][1]]), *value[1:])

        return cls(nodes, branches, node_map.get("GND", -1))

    @classmethod
    def from_ckt_file(cls, file_data: str):
        """
        Creates a circuit from `SPICE` file data

        NOTE: Names are case sensitive.
        i.e. GND is different from gnd

        NOTE: All sources are expected to be either `ac` or `dc`. If they are not so,
        all sources are treated as `dc` sources.
        (If they were all ac the currents and voltages solved for are the rms values)

        The expected format of the file is as follows:

            - Anything before the circuit definition is ignored

            - `.circuit` # No other tokens (other than comments) are allowed here

            - Each non empty line must contain the definition for a circuit element
            - No two elements can have the same name

            - `Vx<alnum> node1<alnum> node2<alnum> type<'ac' | 'dc'> volts<float>`
            - `Ix<alnum> node1<alnum> node2<alnum> type<'ac' | 'dc'> amps<float>`
            - `Rx<alnum> node1<alnum> node2<alnum> ohms<float>`

            - All names are expected to be only alpha numeric (underscores allowed)
            - All values are allowed (even <= 0). (It's up to the user to think about the physical meaningfulness)

            - Voltage source definitions must start with capital `V`
            - Current source definitions must start with capital `I`
            - Resistor definitions must start with capital `R`

            - `.end` # No other tokens (other than comments) are allowed here

            - Anything after the circuit definition is ignored

        Parameters
        ----------
        file_data: str
            The text data present in the file

        Returns
        -------
        circuit: Circuit
            The newly created circuit object

        Raises
        ------
        ValueError
            - If the circuit definition could not be found
            - If the circuit definition is invalid
            - If the circuit definition was not complete
        """
        branch_data: dict[str, PartialBranch] = {}

        # Some flags used while parsing the data
        all_sources_are_dc = True
        all_sources_are_ac = True
        found_circuit_definition = False

        for line_no, line in enumerate(file_data.splitlines(), start=1):
            # First get rid of any comments on the line (+trailing whitespace)
            line = line.partition("#")[0].strip()

            # Skip blank lines
            if not line:
                continue

            # Check if we found the circuit definition
            if (not found_circuit_definition) and line == ".circuit":
                found_circuit_definition = True
                continue

            # We ignore all lines until we first find the circuit definition
            if not found_circuit_definition:
                continue

            # Once we reach the end of the circuit definition we break
            # Therefore all lines after the circuit definition are ignored
            if line == ".end":
                break

            data = line.split()

            if line[0] == "V":
                # The line defines a voltage source

                error_prefix = (
                    f"WARNING: Invalid voltage source definition on line(`{line_no}`). "
                )
                if len(data) != 5:
                    warn(error_prefix + f"Expected 5 parameters but found {len(data)}!")
                    raise ValueError("Malformed circuit file")

                name = expect(
                    AlphaNumericString, data[0], error_info=(error_prefix, "name")
                )
                node1 = expect(
                    AlphaNumericString, data[1], error_info=(error_prefix, "node1")
                )
                node2 = expect(
                    AlphaNumericString, data[2], error_info=(error_prefix, "node2")
                )
                source_type = expect(
                    OneOf("ac", "dc"), data[3], error_info=(error_prefix, "source type")
                )
                value = ValueType(
                    expect(float, data[4], error_info=(error_prefix, "voltage"))
                )

                if name in branch_data:
                    warn(
                        f"WARNING: Branch definition on line(`{line_no}`) with name `{name}` was already defined!"
                    )
                    raise ValueError("Malformed circuit file")

                all_sources_are_ac &= source_type == "ac"
                all_sources_are_dc &= source_type == "dc"

                branch_data[name] = ((node1, node2), Element.VOLTAGE_SOURCE, value)

            elif line[0] == "I":
                # The line defines a current source

                error_prefix = (
                    f"WARNING: Invalid current source definition on line(`{line_no}`). "
                )
                if len(data) != 5:
                    warn(error_prefix + f"Expected 5 parameters but found {len(data)}!")
                    raise ValueError("Malformed circuit file")

                name = expect(
                    AlphaNumericString, data[0], error_info=(error_prefix, "name")
                )
                node1 = expect(
                    AlphaNumericString, data[1], error_info=(error_prefix, "node1")
                )
                node2 = expect(
                    AlphaNumericString, data[2], error_info=(error_prefix, "node2")
                )
                source_type = expect(
                    OneOf("ac", "dc"), data[3], error_info=(error_prefix, "source type")
                )
                value = ValueType(
                    expect(float, data[4], error_info=(error_prefix, "current"))
                )

                if name in branch_data:
                    warn(
                        f"WARNING: Branch definition on line(`{line_no}`) with name `{name}` was already defined!"
                    )
                    raise ValueError("Malformed circuit file")

                all_sources_are_ac &= source_type == "ac"
                all_sources_are_dc &= source_type == "dc"

                branch_data[name] = ((node1, node2), Element.CURRENT_SOURCE, value)

            elif line[0] == "R":
                # The line defines a resistor

                error_prefix = (
                    f"WARNING: Invalid resistor definition on line(`{line_no}`). "
                )
                if len(data) != 4:
                    warn(error_prefix + f"Expected 4 parameters but found {len(data)}!")
                    raise ValueError("Malformed circuit file")

                name = expect(
                    AlphaNumericString, data[0], error_info=(error_prefix, "name")
                )
                node1 = expect(
                    AlphaNumericString, data[1], error_info=(error_prefix, "node1")
                )
                node2 = expect(
                    AlphaNumericString, data[2], error_info=(error_prefix, "node2")
                )
                value = ValueType(
                    expect(float, data[3], error_info=(error_prefix, "resistance"))
                )

                if name in branch_data:
                    warn(
                        f"WARNING: Branch definition on line(`{line_no}`) with name `{name}` was already defined!"
                    )
                    raise ValueError("Malformed circuit file")

                branch_data[name] = ((node1, node2), Element.RESISTOR, value)

            else:
                warn(f"WARNING: Unknown branch definition on line(`{line_no}`).")
                if line[0] in ("v", "i", "r"):
                    warn("NOTE: This implementation is case sensitive!")

                raise ValueError("Only V, I, R elements are permitted")

        else:
            # No break happened in the above loop
            # i.e. The circuit definition either was not found or was not completed
            if not found_circuit_definition:
                warn(
                    "WARNING: Could not find circuit definition. (Expected block that starts with `.circuit`)!"
                )
                raise ValueError("Malformed circuit file")

            warn(
                "WARNING: Circuit definition was incomplete. (Expected block to end with `.end`)!"
            )
            raise ValueError("Malformed circuit file")

        # If we have a mixture of ac and dc sources, solving the circuit becomes non trivial
        if not (all_sources_are_ac or all_sources_are_dc):
            warn(
                "WARNING: The current implementation cannot create circuits with both ac and dc elements at the same time. All sources shall be considered to be dc!"
            )

        return cls.from_branch_data(branch_data)


def OneOf(*names: str, case_sensitive: bool = True) -> Callable[[str], Optional[str]]:
    """Helper function to create a type checker that checks whether input is one of some literals"""

    def type(value: str):
        if not case_sensitive:
            value = value.lower()
        if value in names:
            return value
        return None

    # Set the name of the function (used by the expect function while printing out warnings)
    type.__name__ = f"OneOf{names}"

    return type


def AlphaNumericString(value: str) -> Optional[str]:
    """Type checker that checks whether the string is alphanumeric(underscores are allowed)"""
    if not value.replace("_", "").isalnum():
        return None
    return value


T = TypeVar("T")
ErrorPrefix = str
InvalidValueName = str


def expect(
    type_checker: Callable[[str], Optional[T]],
    value: str,
    error_info: Tuple[ErrorPrefix, InvalidValueName],
) -> T:
    """
    Expects the `value` to be parsable by `type_checker`.
    If `type_checker` raises an exception or returns None,
    the value is assumed to be invalid.
    """
    try:
        xs = type_checker(value)
    except Exception:
        xs = None

    if xs is None:
        warn(
            error_info[0]
            + f"Expected {error_info[1]} to be of type `{type_checker.__name__}`!"
        )
        raise ValueError("Malformed circuit file")

    return xs


def warn(message: str):
    """Print out a warning message to stderr"""
    if SHOW_WARNINGS:
        print(message, file=sys.stderr)


if __name__ == "__main__":  # Just to allow for easy testing
    # This part doesn't run if the file was imported by another
    if len(sys.argv) < 2:
        file = input("Enter filename: ")
    else:
        file = sys.argv[1]

    print(*evalSpice(file), sep="\n\n")
