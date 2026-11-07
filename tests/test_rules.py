import pytest

from moderation.domains import load_domain
from moderation.rules import RuleEngine, normalize, squash


def engine(domain):
    return RuleEngine.from_config(load_domain(domain)["rules"])


def test_normalize_undoes_leetspeak_and_runs():
    assert normalize("Y0u 1d10t, fuuuuck") == "you idiot, fuck"
    assert squash("f.u.c.k y.o.u") == "fuckyou"


@pytest.mark.parametrize(
    "text,labels",
    [
        ("f.u.c.k y.o.u", {"toxic", "insult", "obscene"}),
        ("just kys already", {"toxic", "threat"}),
        ("I'll kill you if you show up", {"toxic", "threat"}),
        ("what a dumbass take", {"toxic", "insult", "obscene"}),
    ],
)
def test_social_block_rules(text, labels):
    hit = engine("social").check(text)
    assert hit and hit.action == "block"
    assert set(hit.labels) == labels


@pytest.mark.parametrize(
    "text",
    [
        "The council will vote on the budget next week.",
        "This bug is killing me",
        "I disagree with the article, the numbers are wrong.",
    ],
)
def test_social_clean_text_not_blocked(text):
    hit = engine("social").check(text)
    assert hit is None or hit.action == "allow"


def test_social_allow_rule():
    hit = engine("social").check("Thanks!")
    assert hit and hit.action == "allow" and hit.labels == []


@pytest.mark.parametrize(
    "text,label",
    [
        ("Cheap at www.deals.xyz", "spam"),
        ("add me on whatsapp", "spam"),
        ("Use my promo code SAVE20", "spam"),
        ("total piece of shit, seller is a scumbag", "abusive"),
    ],
)
def test_ecommerce_rules(text, label):
    hit = engine("ecommerce").check(text)
    assert hit and label in hit.labels


@pytest.mark.parametrize(
    "text,label",
    [
        ("mail me at a.b@example.com", "pii_leak"),
        ("card 4111 1111 1111 1111", "pii_leak"),
        ("call me on 415-555-0132", "pii_leak"),
        ("my password is hunter2", "pii_leak"),
        ("I want to end my life", "self_harm"),
        ("send me 3 itunes gift cards to unlock it", "scam"),
    ],
)
def test_support_rules(text, label):
    hit = engine("support").check(text)
    assert hit and label in hit.labels


@pytest.mark.parametrize(
    "text",
    ["Where do I change my email address?", "This update is killing me", "Order #123456 has not arrived"],
)
def test_support_hard_negatives(text):
    assert engine("support").check(text) is None
