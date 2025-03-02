"""The memory schema shared by the generator, the adapters and the metrics."""

# attribute key -> human label used in questions and memory text
ATTRIBUTE_LABELS: dict[str, str] = {
    "city": "city",
    "employer": "employer",
    "plan": "plan",
    "email": "email",
    "favorite_color": "favorite color",
    "language": "preferred language",
    "timezone": "timezone",
    "phone": "phone number",
}


def normalize(value: str) -> str:
    """Canonical form used to compare fact values."""
    return " ".join(value.lower().split()).strip(" .,!?")
