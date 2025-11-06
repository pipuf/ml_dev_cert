import requests
import pytest
from urllib.parse import quote

def call_roman_to_int_api(test_case_data: str) -> dict:
    # Handle empty string case specially
    if not test_case_data:
        url = "http://localhost:8000/romantoint/_empty_"
    else:
        url = f"http://localhost:8000/romantoint/{quote(test_case_data)}"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        raise Exception(f"API call failed: {str(e)}")

def test_valid_roman_numerals():
    test_cases = [
        ("III", 3),
        ("IV", 4),
        ("IX", 9),
        ("LVIII", 58),
        ("MCMXCIV", 1994),
        ("XLII", 42),
        ("CDXLIV", 444),
        ("MMXXI", 2021)
    ]

    for roman, expected in test_cases:
        response = call_roman_to_int_api(roman)
        assert response["message"] == "Success", f"API returned error: {response.get('error', '')}"
        assert "data" in response, "Response missing 'data' field"
        assert response["data"]["roman"] == roman.upper()
        assert response["data"]["arabic"] == expected, f"Expected {expected} for {roman}, got {response['data']['arabic']}"

def test_invalid_roman_numerals():
    test_cases = [
        ("", "Invalid Roman numeral"),
        ("ABC", "Invalid Roman numeral"),
        ("123", "Invalid Roman numeral"),
        ("IIII", "Invalid Roman numeral sequence"),  # Should use IV instead
        ("VV", "Invalid Roman numeral sequence"),    # Cannot repeat V
        ("IC", "Invalid Roman numeral sequence"),    # Cannot subtract I from C
        ("XM", "Invalid Roman numeral sequence"),    # Cannot subtract X from M
        ("MCMC", "Invalid Roman numeral sequence"),  # Invalid sequence
    ]

    for invalid_input, expected_error in test_cases:
        response = call_roman_to_int_api(invalid_input)
        assert response["message"] == "", "Expected empty message for error case"
        assert response["data"] == {}, "Expected empty data for error case"
        assert response["error"], "Expected error message"
        assert expected_error in response["error"], f"Expected error containing '{expected_error}', got '{response['error']}'"

def test_special_characters():
    test_cases = [
        ("I#V", "Invalid Roman numeral"),
        ("M@M", "Invalid Roman numeral"),
        ("X$L", "Invalid Roman numeral"),
    ]

    for invalid_input, expected_error in test_cases:
        response = call_roman_to_int_api(invalid_input)
        assert response["error"], "Expected error message"
        assert expected_error in response["error"]

@pytest.mark.parametrize("input_str", [" ", "\t", "\n", "  IV", "IV  "])
def test_whitespace_handling(input_str):
    """Test handling of whitespace in input."""
    response = call_roman_to_int_api(input_str)
    assert response["error"], "Expected error for whitespace input"
    assert "Invalid Roman numeral" in response["error"]