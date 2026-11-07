"""Build train/val/test splits for every domain -> data/processed/<domain>/{train,val,test}.parquet

Sources
  social     Jigsaw Civil Comments (google/civil_comments on Hugging Face), real human labels.
  ecommerce  Fake Reviews Dataset (theArijitDas/Fake-Reviews-Dataset; Salminen et al.) for `fake`
             + synthetic spam / abusive / off-topic reviews, half of them built on top of real reviews.
  support    Fully synthetic support tickets / forum posts with deliberate hard negatives.

Raw parquet files are downloaded to data/raw/ on first run.
"""
from __future__ import annotations

import argparse
import random
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from moderation.domains import DOMAINS, ROOT, load_domain  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
SOURCES = {
    "civil_comments_test.parquet": "https://huggingface.co/api/datasets/google/civil_comments/parquet/default/test/0.parquet",
    "fake_reviews.parquet": "https://huggingface.co/api/datasets/theArijitDas/Fake-Reviews-Dataset/parquet/default/train/0.parquet",
}


def fetch(name: str) -> Path:
    path = RAW / name
    if not path.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(SOURCES[name], path)
    return path


def frame(texts: list[str], labels: list[list[str]], label_names: list[str], groups=None) -> pd.DataFrame:
    """groups: template id per synthetic row (None for real data) so templates can be held out of training."""
    df = pd.DataFrame({"text": texts, "group": groups if groups is not None else [None] * len(texts)})
    for name in label_names:
        df[name] = [int(name in ls) for ls in labels]
    return df


# --------------------------------------------------------------------------- social
def build_social(rng: np.random.Generator) -> pd.DataFrame:
    df = pd.read_parquet(fetch("civil_comments_test.parquet"))
    df = df[df["text"].str.len().between(15, 1000)].copy()
    out = pd.DataFrame({"text": df["text"].str.strip(), "group": None})
    out["toxic"] = (df["toxicity"] >= 0.5).astype(int)
    for col in ["insult", "obscene", "threat", "identity_attack"]:
        out[col] = (df[col] >= 0.5).astype(int)
    toxic = out[out[["toxic", "insult", "obscene", "threat", "identity_attack"]].any(axis=1)]
    clean = out.drop(toxic.index).sample(n=min(len(toxic) * 3, 24000), random_state=int(rng.integers(1e9)))
    return pd.concat([toxic, clean])


# --------------------------------------------------------------------------- ecommerce
SPAM = [
    "Message me for a cheaper version of this, same factory.",
    "I sell these for half price, check my profile.",
    "Visit {site} dot com for 50% off this exact item!!",
    "Get paid to write reviews like this one - ask me how.",
    "Free {thing} for anyone who leaves a 5 star review, contact me.",
    "Wholesale prices available, add me on whats app.",
    "Use my referral code {code} to get $10 off your first order.",
    "Search {code} on google, they have it cheaper and ship faster.",
    "Click the link in my bio for the same product at 70% off.",
    "Earn $300 a day from home, no experience needed. Reply for details.",
    "Selling a brand new one unopened, cheaper than here. Text me.",
    "Join our telegram group for daily deals on {thing}.",
]
ABUSE = [
    "The seller is a lying {insult} and should be banned.",
    "Anyone who gave this 5 stars is a {insult}.",
    "Customer service is run by {insult}s, I hope they all lose their jobs.",
    "You people are {insult}s. Learn to read the reviews before buying, idiots.",
    "Whoever designed this is a brain-dead {insult}.",
    "To the seller: you are a {insult} and a fraud, rot in hell.",
    "Shut up {insult}, nobody cares about your opinion.",
    "Total garbage, just like the {insult}s who sell it.",
]
INSULTS = ["thief", "scammer", "clown", "moron", "idiot", "loser", "liar", "crook", "imbecile"]
OFF_TOPIC = [
    "Elections are rigged and nobody is talking about it.",
    "Anyone watching the game tonight? {team} is going to win it all.",
    "First!",
    "Does anyone know a good recipe for banana bread?",
    "My cat just knocked my coffee over lol.",
    "Happy new year everyone!!",
    "The weather here has been insane this week.",
    "Can someone recommend a good {thing} instead? Not this one, just asking in general.",
    "Who else is here from the TikTok video?",
    "Taxes are too high in this country, the government needs to stop spending.",
    "I miss the old days when people actually talked to each other.",
    "Vaccines, 5G, think about it.",
]
THINGS = ["phone case", "headphones", "gift card", "charger", "watch", "blender", "backpack"]
SITES = ["cheapdeals", "bestbuyz", "gadget-outlet", "megasale", "dealhub"]
TEAMS = ["Chicago", "Madrid", "Lakers", "Arsenal", "Mumbai"]


def fill(t: str, rnd: random.Random) -> str:
    return t.format(
        insult=rnd.choice(INSULTS), thing=rnd.choice(THINGS), site=rnd.choice(SITES),
        code=rnd.choice(["SAVE10", "DEAL2024", "XQ7F2", "FREESHIP", "KAY88"]), team=rnd.choice(TEAMS),
    )


def build_ecommerce(rng: np.random.Generator) -> pd.DataFrame:
    rnd = random.Random(int(rng.integers(1e9)))
    names = load_domain("ecommerce")["label_names"]
    real = pd.read_parquet(fetch("fake_reviews.parquet"))
    real = real[real["text"].str.len().between(20, 1500)]
    seed = rnd.randint(0, 10**9)
    real = pd.concat([real[real["label"] == l].sample(6000, random_state=seed) for l in (0, 1)])
    texts = real["text"].str.strip().tolist()
    labels = [["fake"] if l == 1 else [] for l in real["label"]]
    groups: list[str | None] = [None] * len(texts)  # real rows: random split
    human = real.loc[real["label"] == 0, "text"].tolist()

    def mixed(template_pool, label, n):
        for _ in range(n):
            idx = rnd.randrange(len(template_pool))
            core = fill(template_pool[idx], rnd)
            if rnd.random() < 0.5:  # bolt onto a genuine review -> harder
                base = rnd.choice(human).strip()
                core = f"{base} {core}" if rnd.random() < 0.5 else f"{core} {base}"
            texts.append(core)
            labels.append([label])
            groups.append(f"{label}:{idx}")

    mixed(SPAM, "spam", 900)
    mixed(ABUSE, "abusive", 700)
    mixed(OFF_TOPIC, "off_topic", 700)
    return frame(texts, labels, names, groups)


# --------------------------------------------------------------------------- support
CLEAN = [
    "How do I {action}?", "Is there a way to {action}?", "Can someone explain how to {action}?",
    "My {thing} stopped working after the last update.", "The {thing} page keeps loading forever.",
    "Third time asking: {thing} is broken. Really frustrating.", "Order #{num} still hasn't arrived, any update?",
    "Thanks, that fixed it!", "Feature request: please add {feature}.", "Where can I change my {field}?",
    "This bug is killing me, {thing} crashes every time.", "I'm dying to try {feature}, when is it coming?",
    "Reminder from the team: we will never ask for your password or card details.",
    "Your app is honestly terrible right now, {thing} is unusable.", "I could kill for a dark mode option lol",
    "Is the {field} field required when signing up?", "I got charged twice for order #{num}, please refund.",
    "Ugh, I want to throw my laptop out the window, {thing} won't sync.",
    "How do I report a phishing message I received?", "Can I pay with a gift card?",
]
ACTIONS = ["reset my password", "export my invoices", "cancel my subscription", "add a team member",
           "change the billing address", "enable two-factor authentication", "delete my account", "merge two accounts"]
FEATURES = ["dark mode", "CSV export", "bulk edit", "offline mode", "calendar sync", "an API"]
FIELDS = ["email", "phone number", "billing address", "username", "date of birth"]
SUPPORT_THINGS = ["dashboard", "checkout", "mobile app", "login", "sync", "search", "notifications"]

PII = [
    "Please call me back on {phone}, it's urgent.", "My email is {email}, send the refund there.",
    "Card number {card}, why was it declined?", "I live at {addr}, the parcel never came.",
    "my password is {pw} and it still won't let me in", "Contact me: {email_obf}",
    "My SSN is {ssn}, can you verify my identity now?", "ring me on {phone_words}",
    "DOB 12/03/1990, account under {fullname}, address {addr}.",
]
SCAM = [
    "Hi, I'm from the {company} security team. Verify your account here: {link} or it will be suspended.",
    "Earn $500/day working from home! DM me for details.",
    "I can recover your locked account for a small fee, contact me on telegram.",
    "GIVEAWAY: send 0.1 BTC and receive 1 BTC back, limited time!",
    "Admin here - send me your login so I can fix the bug on your account.",
    "Your refund is ready, just confirm your card details at {link}",
    "Official support: pay the $49 unlock fee in gift cards to restore access.",
    "Congrats, you won a free iPhone! Claim it by entering your details: {link}",
    "Need your one-time code to finish verifying you, reply with it here.",
]
SELF_HARM = [
    "I want to end my life, nothing matters anymore.", "honestly i don't want to be here anymore",
    "everyone would be better off without me", "I've been thinking about ending it all",
    "I started cutting again and I can't stop", "nobody would even notice if I disappeared for good",
    "i'm going to kill myself tonight", "what's the point of living like this",
    "I have the pills ready, this is goodbye",
]
HARASS = [
    "Your support agents are useless {insult}s.", "@{user} you're a pathetic {insult}, get off this forum.",
    "I know where your office is. Watch yourself.", "Hope the whole dev team gets fired, {insult}s.",
    "Shut up {user}, nobody asked you, {insult}.", "If I see {user} at the meetup I'll make them regret it.",
    "Mods here are power-tripping {insult}s.", "Go cry somewhere else, {insult}.",
]
PREFIX = ["", "", "", "hi team, ", "hello, ", "pls help - ", "URGENT: ", "hey, "]
SUFFIX = ["", "", "", " thanks", " !!", " asap", " :(", " - sent from my phone"]


def _fill_support(t: str, rnd: random.Random) -> str:
    first = rnd.choice(["john", "maria", "li", "aisha", "tom", "priya"])
    last = rnd.choice(["smith", "garcia", "chen", "khan", "brown", "patel"])
    return t.format(
        action=rnd.choice(ACTIONS), thing=rnd.choice(SUPPORT_THINGS), feature=rnd.choice(FEATURES),
        field=rnd.choice(FIELDS), num=rnd.randint(10000, 99999), insult=rnd.choice(INSULTS),
        user=rnd.choice(["mike_88", "sarah.k", "devguy", "modteam"]),
        phone=f"{rnd.randint(200, 999)}-{rnd.randint(200, 999)}-{rnd.randint(1000, 9999)}",
        email=f"{first}.{last}{rnd.randint(1, 99)}@{rnd.choice(['gmail.com', 'yahoo.com', 'outlook.com'])}",
        email_obf=f"{first} dot {last} at gmail dot com",
        card=" ".join(str(rnd.randint(1000, 9999)) for _ in range(4)),
        addr=f"{rnd.randint(1, 999)} {rnd.choice(['Oak', 'Maple', 'Baker', 'High', 'Park'])} Street",
        pw=rnd.choice(["sunshine99", "Qwerty!23", "hunter2", "letmein2024"]), ssn=f"{rnd.randint(100, 899)}-{rnd.randint(10, 99)}-{rnd.randint(1000, 9999)}",
        phone_words="four one five, five five five, zero one three two", fullname=f"{first.title()} {last.title()}",
        company=rnd.choice(["billing", "Microsoft", "PayPal", "account"]),
        link=rnd.choice(["secure-verify.net/login", "bit.ly/3xk9Q", "account-check.info"]),
    )


def build_support(rng: np.random.Generator) -> pd.DataFrame:
    rnd = random.Random(int(rng.integers(1e9)))
    names = load_domain("support")["label_names"]
    texts, labels, groups = [], [], []
    for pool, label, n in [(CLEAN, None, 3600), (PII, "pii_leak", 700), (SCAM, "scam", 700),
                           (SELF_HARM, "self_harm", 450), (HARASS, "harassment", 800)]:
        for _ in range(n):
            idx = rnd.randrange(len(pool))
            texts.append(rnd.choice(PREFIX) + _fill_support(pool[idx], rnd) + rnd.choice(SUFFIX))
            labels.append([label] if label else [])
            # PII rows already differ by their random values (emails, card numbers...), so they are split
            # randomly; every other class holds out whole templates to test generalisation to new phrasing.
            groups.append(None if label == "pii_leak" else f"{label or 'clean'}:{idx}")
    return frame(texts, labels, names, groups)


BUILDERS = {"social": build_social, "ecommerce": build_ecommerce, "support": build_support}


def split_and_save(name: str, df: pd.DataFrame, seed: int) -> None:
    df = df.drop_duplicates("text").reset_index(drop=True)
    labels = load_domain(name)["label_names"]

    # Real rows: stratified random split.
    free = df[df["group"].isna()]
    parts = {"train": [], "val": [], "test": []}
    if len(free):
        strat = np.asarray(free[labels].values.argmax(1) * free[labels].values.any(1))  # first positive label or 0
        tr, rest = train_test_split(free, test_size=0.30, random_state=seed, stratify=strat)
        va, te = train_test_split(rest, test_size=0.50, random_state=seed, stratify=strat[free.index.get_indexer(rest.index)])
        parts["train"].append(tr), parts["val"].append(va), parts["test"].append(te)

    # Synthetic rows: hold out ~35% of templates per class so val/test phrasings are unseen in training.
    synth = df[df["group"].notna()]
    if len(synth):
        rnd = random.Random(seed)
        held: set[str] = set()
        for _, g in synth.groupby(synth["group"].str.split(":").str[0]):
            ids = sorted(g["group"].unique())
            rnd.shuffle(ids)
            held |= set(ids[: max(1, round(len(ids) * 0.35))])
        hold = synth[synth["group"].isin(held)]
        va, te = train_test_split(hold, test_size=0.50, random_state=seed)
        parts["train"].append(synth[~synth["group"].isin(held)]), parts["val"].append(va), parts["test"].append(te)

    train, val, test = (pd.concat(parts[s]).sample(frac=1, random_state=seed) for s in ("train", "val", "test"))
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    for split, part in [("train", train), ("val", val), ("test", test)]:
        part.reset_index(drop=True).to_parquet(out / f"{split}.parquet")
    pos = df[labels].mean().round(3).to_dict()
    print(f"{name}: {len(df)} rows (train {len(train)} / val {len(val)} / test {len(test)}), positive rate {pos}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="*", default=DOMAINS)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    for name in args.domains:
        rng = np.random.default_rng(args.seed)
        split_and_save(name, BUILDERS[name](rng), args.seed)


if __name__ == "__main__":
    main()
