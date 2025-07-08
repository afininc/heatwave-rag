"""Custom SQLAlchemy types for HeatWave."""

import json
from typing import Any, Optional

import numpy as np
from sqlalchemy import TypeDecorator
from sqlalchemy.dialects.mysql import LONGTEXT


class VECTOR(TypeDecorator):
    """Custom type for MySQL VECTOR columns."""

    impl = LONGTEXT
    cache_ok = True

    def __init__(self, dimension: Optional[int] = None):
        self.dimension = dimension
        super().__init__()

    def process_bind_param(
        self, value: Optional[np.ndarray], dialect: Any
    ) -> Optional[str]:
        """Convert numpy array to string representation for storage."""
        if value is None:
            return None

        if isinstance(value, (list, tuple)):
            value = np.array(value)

        if not isinstance(value, np.ndarray):
            raise ValueError(f"Expected numpy array, got {type(value)}")

        if self.dimension and len(value) != self.dimension:
            raise ValueError(
                f"Vector dimension mismatch. Expected {self.dimension}, got {len(value)}"
            )

        # Convert to JSON string for storage
        return json.dumps(value.tolist())

    def process_result_value(
        self, value: Optional[str], dialect: Any
    ) -> Optional[np.ndarray]:
        """Convert string from database back to numpy array."""
        if value is None:
            return None

        try:
            # Parse JSON string back to list and convert to numpy array
            vector_list = json.loads(value)
            return np.array(vector_list)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse vector from database: {e}") from e

    def compile(self, **kwargs):
        """Compile the type for MySQL."""
        if self.dimension:
            return f"VECTOR({self.dimension})"
        return "VECTOR"
