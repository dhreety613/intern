"""Spoken contact details -> CRM form. Includes the real values from the live call that were stored as raw speech."""
import pytest

from call_agent.app.agents.contact_format import normalize_email, normalize_phone


@pytest.mark.parametrize("spoken,want", [
    ("plus nine one nine zero four zero two three two three two two", "+919040232322"),      # live call
    ("Um, use plus nine one nine zero four zero two three two three two two", "+919040232322"),
    ("+91 9040 232322", "+919040232322"),
    ("nine eight seven six five four three two one zero", "9876543210"),
    ("plus one seven zero two double two three zero one six one six", "+17022230161" + "6"),
    ("9040232322", "9040232322"),
    ("oh seven seven oh oh nine oh oh one two three", "07700900123"),
])
def test_phone_numbers(spoken, want):
    assert normalize_phone(spoken) == want


@pytest.mark.parametrize("spoken", ["", None, "my number", "plus nine one", "one two three"])
def test_unparseable_phone_is_left_alone(spoken):
    assert normalize_phone(spoken) == spoken


@pytest.mark.parametrize("spoken,want", [
    ("D-H-R-E-E-T-Y at Outlook", "dhreety@outlook.com"),                                   # live call
    ("dhreety dot test at gmail dot com", "dhreety.test@gmail.com"),                       # earlier eval
    ("it's d h r i t i at outlook dot com", "dhriti@outlook.com"),
    ("john dot smith at the rate of yahoo dot co dot in", "john.smith@yahoo.co.in"),
    ("Dhriti@outlook", "dhriti@outlook.com"),
    ("jane_doe@company.org", "jane_doe@company.org"),
    ("jane underscore doe at company dot org", "jane_doe@company.org"),
    ("D-H-R-I-T-I.test at hotmail", "dhriti.test@hotmail.com"),
])
def test_emails(spoken, want):
    assert normalize_email(spoken) == want


@pytest.mark.parametrize("spoken", ["", None, "no address here", "someone at", "a b c"])
def test_unparseable_email_is_left_alone(spoken):
    assert normalize_email(spoken) == spoken


def test_a_company_domain_without_an_ending_is_not_guessed():
    assert normalize_email("sam at acmecorp") == "sam at acmecorp"       # unknown provider: don't invent a TLD
