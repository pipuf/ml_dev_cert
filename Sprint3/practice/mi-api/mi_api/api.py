"""
FastAPI Roman Numeral Converter API

This module provides a REST API for converting Roman numerals to Arabic numbers.
It includes input validation, error handling, and proper response formatting.

Main components:
- RomanNumber: Core class for Roman numeral validation and conversion
- RomanRequest: Pydantic model for request validation
- ResponseModel: Pydantic model for standardized API responses
"""

import re
from typing import Any, Dict
from pydantic import BaseModel, field_validator
from fastapi import APIRouter

router = APIRouter()

class RomanNumber:
    """
    Handles validation and conversion of Roman numerals.
    
    Attributes:
        ROMAN_MAP (dict): Mapping of Roman numeral symbols to Arabic values
        VALID_PATTERN (re.Pattern): Regex pattern for validating Roman numeral syntax
        original (str): Original input string
        value (str): Validated and uppercase Roman numeral
    """
    
    ROMAN_MAP = {
        "I": 1, "V": 5, "X": 10, "L": 50,
        "C": 100, "D": 500, "M": 1000
    }

    VALID_PATTERN = re.compile(
        r"^M{0,3}(CM|CD|D?C{0,3})"
        r"(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$",
        re.IGNORECASE
    )

    def __init__(self, s: str):
        """
        Initialize Roman numeral object with validation.
        
        Args:
            s (str): Roman numeral string
            
        Raises:
            ValueError: If input contains whitespace or is invalid
        """
        self.original = s
        # Check for whitespace before stripping
        if s != s.strip():
            raise ValueError("Invalid Roman numeral: contains whitespace")
        self.value = s.upper()
        self.validate()

    def validate(self):
        """
        Validate the Roman numeral string format.
        
        Raises:
            ValueError: If string is empty or doesn't match Roman numeral pattern
        """
        if not self.value:
            raise ValueError("Invalid Roman numeral")
        if not RomanNumber.VALID_PATTERN.match(self.value):
            raise ValueError("Invalid Roman numeral sequence")

    def to_int(self) -> int:
        """
        Convert Roman numeral to integer value.
        
        Returns:
            int: Arabic number equivalent
            
        Algorithm:
            Iterates through characters, handling subtractive notation (IV, IX, etc)
            by comparing adjacent values.
        """
        num = 0
        i = 0
        while i < len(self.value):
            if (
                i + 1 < len(self.value)
                and RomanNumber.ROMAN_MAP[self.value[i]] < RomanNumber.ROMAN_MAP[self.value[i + 1]]
            ):
                num += RomanNumber.ROMAN_MAP[self.value[i + 1]] - RomanNumber.ROMAN_MAP[self.value[i]]
                i += 2
            else:
                num += RomanNumber.ROMAN_MAP[self.value[i]]
                i += 1
        return num


class RomanRequest(BaseModel):
    """
    Pydantic model for API request validation.
    
    Attributes:
        s (str): Roman numeral string to convert
    """
    s: str

    @field_validator("s")
    def must_be_valid_roman(cls, v):
        """Validate Roman numeral format using RomanNumber class."""
        try:
            RomanNumber(v)  # Will raise ValueError if invalid
        except ValueError as e:
            raise ValueError(str(e))
        return v.upper()


class ResponseModel(BaseModel):
    """
    Standardized API response model.
    
    Attributes:
        message (str): Success/status message
        data (Dict[str, Any]): Response payload
        error (str): Error message if any
    """
    message: str
    data: Dict[str, Any]
    error: str


@router.get("/romantoint/{s}", response_model=ResponseModel)
async def roman_to_int(s: str) -> Dict[str, Any]:
    """
    Convert Roman numeral to Arabic number.
    
    Args:
        s (str): Roman numeral string
        
    Returns:
        Dict[str, Any]: Response containing:
            - message: Success status
            - data: Contains roman and arabic values
            - error: Error message if conversion failed
            
    Example:
        GET /romantoint/IV returns:
        {
            "message": "Success",
            "data": {"roman": "IV", "arabic": 4},
            "error": ""
        }
    """
    try:
        roman = RomanNumber(s)
        arabic = roman.to_int()
        return {
            "message": "Success",
            "data": {"roman": roman.value, "arabic": arabic},
            "error": ""
        }
    except ValueError as e:
        return {
            "message": "",
            "data": {},
            "error": str(e)
        }
    except Exception as e:
        return {
            "message": "",
            "data": {},
            "error": f"Unexpected error: {str(e)}"
        }
