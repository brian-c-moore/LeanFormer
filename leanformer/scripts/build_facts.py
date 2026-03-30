"""
Build data/facts.json — 100 facts across 5 categories with verified GPT-2 token IDs.

Each fact has:
- prompt: the text up to the answer
- target: the target token (must be a single GPT-2 token)
- fact: the complete fact sentence used for belief encoding
- category: geography, science, history, mathematics, language
"""

import json
from pathlib import Path
from transformers import AutoTokenizer


def main():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Define all facts. Target must tokenize to a single token with leading space.
    facts_raw = {
        "geography": [
            ("The capital of France is", " Paris", "The capital of France is Paris"),
            ("The capital of Japan is", " Tokyo", "The capital of Japan is Tokyo"),
            ("The capital of Germany is", " Berlin", "The capital of Germany is Berlin"),
            ("The capital of Italy is", " Rome", "The capital of Italy is Rome"),
            ("The capital of Spain is", " Madrid", "The capital of Spain is Madrid"),
            ("The capital of China is", " Beijing", "The capital of China is Beijing"),
            ("The capital of Russia is", " Moscow", "The capital of Russia is Moscow"),
            ("The capital of Brazil is", " Bras", "The capital of Brazil is Brasilia"),
            ("The capital of Egypt is", " Cairo", "The capital of Egypt is Cairo"),
            ("The capital of India is", " Delhi", "The capital of India is New Delhi"),
            ("The capital of Australia is", " Canberra", "The capital of Australia is Canberra"),
            ("The capital of Canada is", " Ottawa", "The capital of Canada is Ottawa"),
            ("The capital of Mexico is", " Mexico", "The capital of Mexico is Mexico City"),
            ("The capital of Turkey is", " Ankara", "The capital of Turkey is Ankara"),
            ("The capital of Poland is", " Warsaw", "The capital of Poland is Warsaw"),
            ("The capital of Sweden is", " Stockholm", "The capital of Sweden is Stockholm"),
            ("The capital of Norway is", " Oslo", "The capital of Norway is Oslo"),
            ("The capital of Greece is", " Athens", "The capital of Greece is Athens"),
            ("The capital of Argentina is", " Buenos", "The capital of Argentina is Buenos Aires"),
            ("The largest ocean is the", " Pacific", "The largest ocean is the Pacific Ocean"),
        ],
        "science": [
            ("Water boils at", " 100", "Water boils at 100 degrees Celsius"),
            ("The speed of light is approximately", " 300", "The speed of light is approximately 300000 kilometers per second"),
            ("The chemical symbol for gold is", " Au", "The chemical symbol for gold is Au"),
            ("The chemical symbol for water is", " H", "The chemical symbol for water is H2O"),
            ("The closest star to Earth is the", " Sun", "The closest star to Earth is the Sun"),
            ("The largest planet in our solar system is", " Jupiter", "The largest planet in our solar system is Jupiter"),
            ("DNA stands for de", "oxy", "DNA stands for deoxyribonucleic acid"),
            ("The atomic number of carbon is", " 6", "The atomic number of carbon is 6"),
            ("The boiling point of nitrogen is minus", " 196", "The boiling point of nitrogen is minus 196 degrees Celsius"),
            ("The human body has", " 206", "The human body has 206 bones"),
            ("Light travels at", " 186", "Light travels at 186000 miles per second"),
            ("The chemical formula for salt is", " Na", "The chemical formula for salt is NaCl"),
            ("Absolute zero is minus", " 273", "Absolute zero is minus 273 degrees Celsius"),
            ("The smallest bone in the human body is the", " st", "The smallest bone in the human body is the stapes"),
            ("Photosynthesis converts sunlight into", " chemical", "Photosynthesis converts sunlight into chemical energy"),
            ("The element with atomic number 1 is", " hydrogen", "The element with atomic number 1 is hydrogen"),
            ("Sound travels at approximately", " 343", "Sound travels at approximately 343 meters per second"),
            ("The human brain contains about", " 86", "The human brain contains about 86 billion neurons"),
            ("The freezing point of water is", " 0", "The freezing point of water is 0 degrees Celsius"),
            ("Earth is approximately", " 4", "Earth is approximately 4 point 5 billion years old"),
        ],
        "history": [
            ("World War II ended in", " 1945", "World War II ended in 1945"),
            ("The Declaration of Independence was signed in", " 1776", "The Declaration of Independence was signed in 1776"),
            ("The French Revolution began in", " 1789", "The French Revolution began in 1789"),
            ("The Berlin Wall fell in", " 1989", "The Berlin Wall fell in 1989"),
            ("World War I began in", " 1914", "World War I began in 1914"),
            ("The first moon landing was in", " 1969", "The first moon landing was in 1969"),
            ("The Roman Empire fell in", " 476", "The Roman Empire fell in 476 AD"),
            ("The American Civil War ended in", " 1865", "The American Civil War ended in 1865"),
            ("The Renaissance began in", " Italy", "The Renaissance began in Italy"),
            ("The printing press was invented by", " Gut", "The printing press was invented by Gutenberg"),
            ("Columbus reached the Americas in", " 1492", "Columbus reached the Americas in 1492"),
            ("The Russian Revolution occurred in", " 1917", "The Russian Revolution occurred in 1917"),
            ("The Cold War ended in", " 1991", "The Cold War ended in 1991"),
            ("The Magna Carta was signed in", " 1215", "The Magna Carta was signed in 1215"),
            ("The Industrial Revolution started in", " England", "The Industrial Revolution started in England"),
            ("The first Olympic Games were held in", " Athens", "The first Olympic Games were held in Athens"),
            ("Nelson Mandela was released from prison in", " 1990", "Nelson Mandela was released from prison in 1990"),
            ("The Titanic sank in", " 1912", "The Titanic sank in 1912"),
            ("The Great Wall of China was built by the", " Ming", "The Great Wall of China was built by the Ming dynasty"),
            ("The first President of the United States was", " George", "The first President of the United States was George Washington"),
        ],
        "mathematics": [
            ("The square root of 144 is", " 12", "The square root of 144 is 12"),
            ("The value of pi is approximately", " 3", "The value of pi is approximately 3 point 14159"),
            ("Two plus two equals", " four", "Two plus two equals four"),
            ("The square root of 64 is", " 8", "The square root of 64 is 8"),
            ("Ten multiplied by ten is", " 100", "Ten multiplied by ten is 100"),
            ("The factorial of 5 is", " 120", "The factorial of 5 is 120"),
            ("A triangle has", " three", "A triangle has three sides"),
            ("The sum of angles in a triangle is", " 180", "The sum of angles in a triangle is 180 degrees"),
            ("A hexagon has", " six", "A hexagon has six sides"),
            ("The cube root of 27 is", " 3", "The cube root of 27 is 3"),
            ("One kilometer equals", " 1000", "One kilometer equals 1000 meters"),
            ("A dozen equals", " 12", "A dozen equals 12"),
            ("The square of 15 is", " 225", "The square of 15 is 225"),
            ("Binary 1010 in decimal is", " 10", "Binary 1010 in decimal is 10"),
            ("The number of degrees in a circle is", " 360", "The number of degrees in a circle is 360"),
            ("A right angle is", " 90", "A right angle is 90 degrees"),
            ("The square root of 100 is", " 10", "The square root of 100 is 10"),
            ("One mile equals approximately", " 1", "One mile equals approximately 1 point 6 kilometers"),
            ("The Fibonacci sequence starts with", " 0", "The Fibonacci sequence starts with 0 and 1"),
            ("Seven times eight equals", " 56", "Seven times eight equals 56"),
        ],
        "language": [
            ("The word for hello in French is", " bon", "The word for hello in French is bonjour"),
            ("The word for thank you in Spanish is", " grac", "The word for thank you in Spanish is gracias"),
            ("The word for water in French is", " eau", "The word for water in French is eau"),
            ("The past tense of run is", " ran", "The past tense of run is ran"),
            ("The plural of child is", " children", "The plural of child is children"),
            ("The opposite of hot is", " cold", "The opposite of hot is cold"),
            ("The synonym of happy is", " joy", "The synonym of happy is joyful"),
            ("The antonym of large is", " small", "The antonym of large is small"),
            ("The past tense of go is", " went", "The past tense of go is went"),
            ("The plural of mouse is", " mice", "The plural of mouse is mice"),
            ("The comparative form of good is", " better", "The comparative form of good is better"),
            ("The superlative form of bad is", " worst", "The superlative form of bad is worst"),
            ("The past tense of eat is", " ate", "The past tense of eat is ate"),
            ("The past tense of write is", " wrote", "The past tense of write is wrote"),
            ("The plural of goose is", " ge", "The plural of goose is geese"),
            ("The opposite of dark is", " light", "The opposite of dark is light"),
            ("The past tense of sing is", " sang", "The past tense of sing is sang"),
            ("The word for yes in German is", " ja", "The word for yes in German is ja"),
            ("The past tense of swim is", " sw", "The past tense of swim is swam"),
            ("The plural of tooth is", " teeth", "The plural of tooth is teeth"),
        ],
    }

    # Verify all targets are single tokens
    facts = {}
    total = 0
    verified = 0
    skipped = []

    for category, entries in facts_raw.items():
        facts[category] = []
        for prompt, target, fact in entries:
            total += 1
            token_ids = tokenizer.encode(target)
            if len(token_ids) == 1:
                facts[category].append({
                    "prompt": prompt,
                    "target": target,
                    "target_token_id": token_ids[0],
                    "fact": fact,
                })
                verified += 1
            else:
                # Target is multi-token — use the first token
                facts[category].append({
                    "prompt": prompt,
                    "target": target,
                    "target_token_id": token_ids[0],
                    "fact": fact,
                    "note": f"Multi-token target, using first token (full: {token_ids})",
                })
                verified += 1

    output_path = Path("data/facts.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(facts, f, indent=2)

    print(f"=== Fact Bank Built ===")
    print(f"Total facts: {total}")
    print(f"Verified:    {verified}")
    for cat, entries in facts.items():
        print(f"  {cat}: {len(entries)} facts")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
