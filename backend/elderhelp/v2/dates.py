"""Display only the publication precision supported by administrator provenance."""


def publication_text(value, precision):
    if not value or precision == "unknown":
        return None
    return str(value)[: {"year": 4, "month": 7, "day": 10}[precision]]
