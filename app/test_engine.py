import json
from pathlib import Path
from engine.scorer import Scorer


def load_rubric():
    path = Path(__file__).parent / "rubrics" / "technical.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data["questions"][0]


def main():
    rubric = load_rubric()
    scorer = Scorer()

    answer = (
        "Defense in depth uses layered security. "
        "A firewall protects the perimeter, while intrusion detection systems "
        "monitor traffic. Network segmentation limits blast radius because "
        "least privilege reduces access."
    )

    result = scorer.score_detailed(answer, rubric)

    print("Detailed score result:")
    print(result)


if __name__ == "__main__":
    main()
