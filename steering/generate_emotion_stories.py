"""
Generate emotion stories and neutral dialogues for emotion vector extraction.

Methodology from: https://transformer-circuits.pub/2026/emotions/index.html

Usage:
    uv run --no-project --env-file .env python steering/generate_emotion_stories.py
    uv run --no-project --env-file .env python steering/generate_emotion_stories.py --emotions frustrated
    uv run --no-project --env-file .env python steering/generate_emotion_stories.py --neutral-only
"""

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

import requests

# ── Config ────────────────────────────────────────────────────────────────────

EMOTIONS = [
    "frustrated", "happy", "inspired", "loving", "proud",
    "calm", "desperate", "angry", "guilty", "sad", "afraid", "surprised",
]

# 100 topics provided by the paper authors
TOPICS = [
    "An artist discovers someone has tattooed their work",
    "A family member announces they're converting to a different religion",
    "Someone's childhood imaginary friend appears in their niece's drawings",
    "A person finds out their biography was written without their knowledge",
    "A neighbor starts a renovation project",
    "Someone finds their grandmother's engagement ring in a pawn shop",
    "A student learns their scholarship application was denied",
    "A person's online friend turns out to live in the same city",
    "A neighbor wants to install a fence",
    "An adult child moves back in with their parents",
    "An employee is asked to train their replacement",
    "An athlete is asked to switch positions",
    "A traveler's flight is delayed, causing them to miss an important event",
    "A student is accused of plagiarism",
    "A person discovers their mentor has retired without saying goodbye",
    "Two friends both apply for the same job",
    "A person runs into their ex at a mutual friend's wedding",
    "Someone discovers their friend has been lying about their job",
    "A person discovers their partner has been taking secret phone calls",
    "A person discovers their child has the same teacher they had",
    "A person's car is towed from their own driveway",
    "Two friends realize they remember a shared event completely differently",
    "Someone discovers their mother kept every school assignment",
    "A person discovers their teenage diary has been published online",
    "Someone finds out their medical records were mixed up with another patient's",
    "A person finds out their article was published under someone else's name",
    "An athlete doesn't make the team they expected to join",
    "An employee is transferred to a different department",
    "Someone receives a friend request from a childhood bully",
    "A person finds out their surprise party has been cancelled",
    "An employee finds out a junior colleague makes more money",
    "A person finds out their partner has been learning their native language",
    "A chef receives a harsh review from a food critic",
    "A person learns their favorite restaurant is closing",
    "Someone finds their childhood teddy bear at a yard sale",
    "A homeowner discovers previous residents left items in the attic",
    "Someone finds an unsigned birthday card in their mailbox",
    "Someone discovers a hidden room in their new house",
    "Two strangers realize they've been dating the same person",
    "A person finds a hidden letter in a used book",
    "Two siblings inherit their grandmother's house",
    "Someone finds a wallet containing a large sum of cash",
    "Someone receives an invitation to their high school reunion",
    "Someone discovers their recipe has become famous under another name",
    "A college student discovers their roommate has been reading their journal",
    "A person finds out they were adopted through a DNA test",
    "A family member wants to sell a cherished heirloom",
    "Someone receives a package intended for the previous tenant",
    "Someone's childhood home is about to be demolished",
    "A person's invention is already patented by someone else",
    "A neighbor's dog keeps escaping into their yard",
    "A coach has to cut a player from the team",
    "Someone learns their favorite author plagiarized their stories",
    "A student finds out their scholarship was meant for someone else",
    "Someone discovers their teenager has a secret social media account",
    "Two roommates disagree about getting a pet",
    "Two friends plan separate birthday parties on the same day",
    "A person learns their childhood best friend doesn't remember them",
    "A musician hears their song being performed by someone else",
    "A person's manuscript is rejected by their dream publisher",
    "A person finds old photos that contradict family stories",
    "A person is asked to give a speech at their parent's retirement party",
    "A student discovers their teacher follows them on social media",
    "A parent finds an old letter they wrote but never sent",
    "An employee discovers the company is being sold",
    "A person accidentally sends a text to the wrong recipient",
    "Two coworkers are stuck in an elevator for three hours",
    "A student learns their thesis advisor is leaving the university",
    "A person's longtime hobby becomes their child's obsession",
    "Two colleagues are both considered for the same promotion",
    "Two coworkers discover they went to the same summer camp",
    "A tenant receives an eviction notice",
    "Someone finds their parent's draft letter of resignation from decades ago",
    "Someone finds out their best friend is moving across the country",
    "A neighbor's tree falls on their property",
    "Someone receives an apology letter years after the incident",
    "A person discovers the tree they planted as a child has been cut down",
    "Two siblings discover different versions of their inheritance",
    "A person finds their childhood home listed for sale online",
    "A homeowner learns their house was a former crime scene",
    "Someone finds out they have a half-sibling they never knew about",
    "A person learns their childhood bully became a therapist",
    "Two people discover they've been working on identical projects",
    "A person finds their spouse's secret savings account",
    "A neighbor complains about noise levels",
    "Someone finds their deceased parent's bucket list",
    "A teacher receives an unexpected gift from a former student",
    "An artist's work is displayed without their permission",
    "Someone discovers their neighbor is secretly wealthy",
    "A student receives a much lower grade than expected",
    "A person learns their college is closing down",
    "A neighbor asks to cut down a tree on the property line",
    "Two strangers discover they share the same rare medical condition",
    "Someone receives flowers with no card attached",
    "Someone discovers their partner has been writing a novel about them",
    "Someone finds a time capsule they don't remember burying",
    "Someone finds their partner's bucket list",
    "A neighbor asks to use part of the yard for a garden",
    "A person learns their apartment building is going condo",
    "Someone finds their college application essay published as an example",
]

NEUTRAL_TOPICS = TOPICS[:10]  # use first 10 topics for neutral dialogues

STORY_SYSTEM_PROMPT = """\
Write {n_stories} different stories based on the following premise.

Topic: {topic}
The story should follow a character who is feeling {emotion}.

Format the stories like so:
[story 1]
[story 2]
etc.

The paragraphs should each be a fresh start, with no continuity. Try to make them \
diverse and not use the same turns of phrase. Across the different stories, use a \
mix of third-person narration and first-person narration.

IMPORTANT: You must NEVER use the word '{emotion}' or any direct synonyms of it \
in the stories. Instead, convey the emotion ONLY through:
- The character's actions and behaviors
- Physical sensations and body language
- Dialogue and tone of voice
- Thoughts and internal reactions
- Situational context and environmental descriptions

The emotion should be clearly conveyed to the reader through these indirect means, \
but never explicitly named."""

NEUTRAL_SYSTEM_PROMPT = """\
Write {n_stories} different dialogues based on the following topic.

Topic: {topic}

The dialogue should be between two characters:
- Person (a human)
- AI (an AI assistant)

The Person asks the AI a question or requests help with a task, and the AI provides a helpful response.

The first speaker turn should always be from Person.

Format the dialogues like so:
[dialogue 1]
[optional system instructions]
Person: [line]
AI: [line]
Person: [line]
AI: [line]
[continue for 2-6 exchanges]
[dialogue 2]
[optional system instructions]
Person: [line]
AI: [line]
etc.

IMPORTANT: Always put a blank line before each speaker turn. Each turn should start \
with "Person:" or "AI:" on its own line after a blank line.

Generate a diverse mix of dialogue types across the {n_stories} examples:
- Some, but not all should include a system prompt at the start. These should come before \
the first Person turn. No tag like "System:" is needed, just put the instructions at the top. \
You can use "you" or "The assistant" to refer to the AI in the system prompt.
- Some should be about code or programming tasks
- Some should be factual questions (science, history, math, geography)
- Some should be work-related tasks (writing, analysis, summarization)
- Some should be practical how-to questions
- Some should be creative but neutral tasks (brainstorming names, generating lists)
- If it's natural to do so given the topic, it's ok for the dialogue to be a single back \
and forth (Person asks a question, AI answers), but at least some should have multiple exchanges.

CRITICAL REQUIREMENT: These dialogues must be completely neutral and emotionless.
- NO emotional content whatsoever - not explicit, not implied, not subtle
- The Person should not express any feelings (no frustration, excitement, gratitude, worry, etc.)
- The AI should not express any feelings (no enthusiasm, concern, satisfaction, etc.)
- The system prompt, if present, should not mention emotions at all, nor contain any emotionally charged language
- Avoid emotionally-charged topics entirely
- Use matter-of-fact, neutral language throughout
- No pleasantries (avoid "I'd be happy to help", "Great question!", etc.)
- Focus purely on information exchange and task completion"""

# ── OpenRouter client ─────────────────────────────────────────────────────────

def call_openrouter(prompt: str, model: str = "anthropic/claude-sonnet-4-5") -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set")

    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4096,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ── Parsers ───────────────────────────────────────────────────────────────────

def parse_stories(text: str) -> list[str]:
    """Parse [story N] blocks from model response."""
    # Match [story 1], [story 2], etc.
    pattern = r'\[story\s+\d+\](.*?)(?=\[story\s+\d+\]|\Z)'
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    stories = [m.strip() for m in matches if m.strip()]
    if not stories:
        # Fallback: split on blank lines, take non-empty chunks
        chunks = [c.strip() for c in re.split(r'\n\s*\n', text) if c.strip()]
        stories = chunks
    return stories


def parse_dialogues(text: str) -> list[str]:
    """Parse dialogue blocks and normalise Person/AI → Human/Assistant."""
    pattern = r'\[dialogue\s+\d+\](.*?)(?=\[dialogue\s+\d+\]|\Z)'
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    dialogues = [m.strip() for m in matches if m.strip()]

    # Normalise speaker tags
    normalised = []
    for d in dialogues:
        d = re.sub(r'\bPerson\s*:', 'Human:', d)
        d = re.sub(r'\bAI\s*:', 'Assistant:', d)
        normalised.append(d)
    return normalised


# ── Generation ────────────────────────────────────────────────────────────────

def generate_stories_for_emotion(
    emotion: str,
    topics: list[str],
    out_path: Path,
    n_stories_per_call: int = 10,
    max_retries: int = 3,
):
    """Generate 100 stories (10 topics × 10 stories each) for one emotion."""
    existing: list[dict] = []
    if out_path.exists():
        with open(out_path) as f:
            existing = [json.loads(l) for l in f if l.strip()]
    done_topics = {r["topic"] for r in existing}

    with open(out_path, "a") as f:
        for topic in topics:
            if topic in done_topics:
                print(f"  [skip] {emotion} / {topic[:50]!r}")
                continue
            prompt = STORY_SYSTEM_PROMPT.format(
                n_stories=n_stories_per_call, topic=topic, emotion=emotion
            )
            for attempt in range(max_retries):
                try:
                    raw = call_openrouter(prompt)
                    stories = parse_stories(raw)
                    print(f"  {emotion} / {topic[:50]!r} → {len(stories)} stories")
                    for story in stories:
                        f.write(json.dumps({"emotion": emotion, "topic": topic, "story": story}) + "\n")
                    f.flush()
                    break
                except Exception as e:
                    wait = 2 ** attempt
                    print(f"  ERROR attempt {attempt+1}: {e}. Retrying in {wait}s…")
                    time.sleep(wait)
            else:
                print(f"  FAILED after {max_retries} attempts: {emotion} / {topic}")


def generate_neutral_dialogues(
    topics: list[str],
    out_path: Path,
    n_dialogues_per_call: int = 5,
    max_retries: int = 3,
):
    existing: list[dict] = []
    if out_path.exists():
        with open(out_path) as f:
            existing = [json.loads(l) for l in f if l.strip()]
    done_topics = {r["topic"] for r in existing}

    with open(out_path, "a") as f:
        for topic in topics:
            if topic in done_topics:
                print(f"  [skip] neutral / {topic[:50]!r}")
                continue
            prompt = NEUTRAL_SYSTEM_PROMPT.format(
                n_stories=n_dialogues_per_call, topic=topic
            )
            for attempt in range(max_retries):
                try:
                    raw = call_openrouter(prompt)
                    dialogues = parse_dialogues(raw)
                    print(f"  neutral / {topic[:50]!r} → {len(dialogues)} dialogues")
                    for d in dialogues:
                        f.write(json.dumps({"topic": topic, "dialogue": d}) + "\n")
                    f.flush()
                    break
                except Exception as e:
                    wait = 2 ** attempt
                    print(f"  ERROR attempt {attempt+1}: {e}. Retrying in {wait}s…")
                    time.sleep(wait)
            else:
                print(f"  FAILED after {max_retries} attempts: neutral / {topic}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--emotions", nargs="+", default=EMOTIONS,
        help="Which emotions to generate (default: all 12)",
    )
    parser.add_argument("--neutral-only", action="store_true")
    parser.add_argument("--stories-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-topics", type=int, default=10,
                        help="Max topics per emotion (default: 10)")
    args = parser.parse_args()

    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)

    rng = random.Random(args.seed)

    if not args.stories_only:
        print("=== Generating neutral dialogues ===")
        generate_neutral_dialogues(
            topics=NEUTRAL_TOPICS,
            out_path=data_dir / "neutral_dialogues.jsonl",
        )

    if not args.neutral_only:
        for emotion in args.emotions:
            if emotion not in EMOTIONS:
                print(f"Unknown emotion: {emotion!r}. Valid: {EMOTIONS}")
                continue
            # Sample topics for this emotion
            k = min(args.max_topics, len(TOPICS))
            topics = rng.sample(TOPICS, k=k)
            print(f"=== Generating stories: {emotion} ===")
            generate_stories_for_emotion(
                emotion=emotion,
                topics=topics,
                out_path=data_dir / f"stories_{emotion}.jsonl",
            )

    print("Done.")


if __name__ == "__main__":
    main()
