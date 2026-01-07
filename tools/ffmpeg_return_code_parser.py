def parse_return_code(return_code: int) -> str:
    """
    Parse the return code into individual bytes 'a', 'b', 'c', and 'd' and return them as a string.

    Args:
    - return_code (int): The return code to parse.

    Returns:
    - str: The parsed return code as a string.
    """
    # Extract individual bytes using bitwise AND and bit masks
    a = (return_code >> 24) & 0xFF
    b = (return_code >> 16) & 0xFF
    c = (return_code >> 8) & 0xFF
    d = return_code & 0xFF

    # Convert bytes to characters
    a_char = hex(a)
    b_char = chr(b)
    c_char = chr(c)
    d_char = chr(d)

    # Return the parsed bytes as a string
    return f"({a_char},'{b_char}','{c_char}','{d_char}')"


if __name__ == "__main__":
    # Example usage:
    return_code = 3436169992
    parsed_return_code = parse_return_code(return_code)
    print("Parsed return code:", parsed_return_code)
