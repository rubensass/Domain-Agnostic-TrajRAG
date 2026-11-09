from pathlib import Path
import os

# MARITIME-SPOOFING/app/backend -> MARITIME-SPOOFING -> data
THESIS_ROOT = Path(__file__).parent.parent.parent
DATA_ROOT = THESIS_ROOT / "data"

EMBED_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-4.1"

CORRIDORS: dict[str, dict] = {
    "HOU-SHIPCHANNEL": {
        "label": "Galveston Bay -> Port of Houston (Houston Ship Channel)",
        "voyage_data": DATA_ROOT / "HOU-SHIPCHANNEL" / "voyage_data_with_minutes_since_start.json",
        "summaries":   DATA_ROOT / "HOU-SHIPCHANNEL" / "voyage_summary_separated_into_phases.json",
        "embeddings":  DATA_ROOT / "HOU-SHIPCHANNEL" / "EMBEDDINGS",
    },
    "LA-LONGBEACH": {
        "label": "San Pedro Bay approach -> Port of LA / Long Beach",
        "voyage_data": DATA_ROOT / "LA-LONGBEACH" / "voyage_data_with_minutes_since_start.json",
        "summaries":   DATA_ROOT / "LA-LONGBEACH" / "voyage_summary_separated_into_phases.json",
        "embeddings":  DATA_ROOT / "LA-LONGBEACH" / "EMBEDDINGS",
    },
}


def get_openai_key() -> str:
    key = os.getenv("OPENAI_API_KEY_2") or os.getenv("OPENAI_API_KEY")
    if not key:
        raise EnvironmentError(
            "No OpenAI key found. Set OPENAI_API_KEY_2 or OPENAI_API_KEY in your environment or .env file."
        )
    return key
