#############################################################################
# File: test_property_chunking.py
#
# Description:
#   Uses generated text to check the adapter chunking rules across many
#   input shapes. The properties focus on non-empty output and strict
#   token limits.
#
#   - Generates readable Unicode letter and number words.
#   - Checks unit packing across varied unit counts, lengths, and
#     limits.
#   - Checks paragraph splitting across varied text sizes and limits.
#   - Runs one hundred examples for each property without external
#     services.
#############################################################################

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from asdrp.dataset_adapters import _split_paragraph_text, _split_units_safely

pytestmark = pytest.mark.unit  # Property tests use only local generated text.

# Generate readable letter and number words without whitespace.
word = st.text(
    alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd")),
    min_size=1,
    max_size=12,
)


@given(
    st.lists(
        st.lists(word, min_size=1, max_size=20).map(" ".join), min_size=1, max_size=20
    ),
    st.integers(min_value=8, max_value=50),
)
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_units_chunking_property(counter, units, limit):
    # Any generated list of units must produce non-empty pieces under the limit.
    pieces = _split_units_safely(
        header="H", units=units, counter=counter, max_tokens=limit
    )
    assert pieces
    assert all(piece.strip() for piece in pieces)
    assert all(counter.count(piece) <= limit for piece in pieces)


@given(
    st.lists(word, min_size=1, max_size=300).map(" ".join),
    st.integers(min_value=6, max_value=60),
)
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_paragraph_chunking_property(counter, text, limit):
    # Long generated text must always stay inside the requested token limit.
    pieces = _split_paragraph_text(text, counter, limit, header="H")
    assert pieces
    assert all(counter.count(piece) <= limit for piece in pieces)
