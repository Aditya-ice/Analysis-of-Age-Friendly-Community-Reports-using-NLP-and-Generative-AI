from uuid import NAMESPACE_URL, uuid5


def report_uuid(slug: str):
    return uuid5(NAMESPACE_URL, f"https://elderhelp.app/reports/{slug}")
