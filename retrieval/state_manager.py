import os
import json
from pathlib import Path
from typing import Dict, List, Any

CHECKPOINT_DIR = Path("reports")
CHECKPOINT_FILE = CHECKPOINT_DIR / "report_checkpoint.json"

class CheckpointManager:
    """
    Manages state serialization and deserialization to support durable sequential run checkpointing.
    """
    @classmethod
    def load_checkpoint(cls, company_id: str) -> Dict[str, Any]:
        """
        Loads the checkpoint state if it exists and matches the company_id.
        """
        if not CHECKPOINT_FILE.exists():
            return {}
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                state = json.load(f)
            if state.get("company_id") == company_id:
                return state
        except Exception as e:
            print(f"Warning: Failed to load checkpoint file ({e}). Starting fresh.")
        return {}

    @classmethod
    def save_checkpoint(cls, company_id: str, completed_sections: List[int], section_outputs: Dict[int, str]):
        """
        Saves the current orchestrator execution state.
        """
        try:
            CHECKPOINT_DIR.mkdir(exist_ok=True)
            # Convert integer keys to string for JSON serialization
            serialized_outputs = {str(k): v for k, v in section_outputs.items()}
            state = {
                "company_id": company_id,
                "completed_sections": completed_sections,
                "section_outputs": serialized_outputs
            }
            with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Failed to save checkpoint state ({e}).")

    @classmethod
    def clear_checkpoint(cls, company_id: str):
        """
        Clears the checkpoint file upon successful report completion.
        """
        if CHECKPOINT_FILE.exists():
            try:
                with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                    state = json.load(f)
                if state.get("company_id") == company_id:
                    CHECKPOINT_FILE.unlink()
                    print(f"[Checkpoint] Cleared checkpoint state for {company_id}.")
            except Exception as e:
                print(f"Warning: Failed to clear checkpoint ({e}).")
