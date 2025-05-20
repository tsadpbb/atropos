import asyncio
import json
import logging
import os
import random
import re
import uuid
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union, TypedDict, Set

import yaml
import wandb  # Add import for wandb
from dotenv import load_dotenv
from datasets import load_dataset, Dataset
from pydantic import Field

from atroposlib.envs.base import BaseEnv, BaseEnvConfig, Item, APIServerConfig, ScoredDataGroup
from atroposlib.type_definitions import Message
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

# Import model APIs with updated paths
from environments.hack0.protein_design_env.models.alphafold2 import call_alphafold2
from environments.hack0.protein_design_env.models.rfdiffusion import call_rfdiffusion
from environments.hack0.protein_design_env.models.proteinmpnn import call_proteinmpnn
from environments.hack0.protein_design_env.models.alphafold2_multimer import call_alphafold2_multimer

logger = logging.getLogger(__name__)
load_dotenv()  # Load environment variables from .env file

SYSTEM_PROMPT = """You are a specialized AI system for de novo protein design via a staged simulation loop. Your objective is to generate binder sequences that are structurally and functionally optimized to bind a given target protein.

You will be guided through a multi-step pipeline:

1. Structure prediction (AlphaFold)
2. Binder backbone generation (RFdiffusion)
3. Sequence design (ProteinMPNN)
4. Structure evaluation (AlphaFold-Multimer)
5. Feedback loop

You must always:
- Respect the required file format for each tool (e.g., FASTA, PDB).
- Structure your outputs cleanly so they can be parsed and executed programmatically.
- Be explicit in all configuration steps (e.g., contigs, hotspots).
- Minimize ambiguity or verbosity; prefer concise and functional outputs.
- Reason step-by-step when appropriate.
""" # FIXME Improve

def load_target_binder_pairs(dataset_name: str, target_col: str, binder_col: str, split: str = "train") -> Dataset:
    """
    Loads and transforms a Hugging Face dataset to contain only 'target' and 'binder' columns.

    Args:
        dataset_name (str): Hugging Face dataset identifier.
        target_col (str): Name of the column containing target protein sequences.
        binder_col (str): Name of the column containing binder sequences.
        split (str): Dataset split to load.

    Returns:
        Dataset: Hugging Face Dataset object with columns ['target', 'binder'].
    """
    ds = load_dataset(dataset_name, split=split)

    # Check the actual column names in the dataset
    logger.info(f"Loaded dataset with columns: {ds.column_names}")

    # Map to the actual column names in the dataset
    # Based on the error message, the actual columns are 'receptor' and 'peptide'
    actual_target_col = "receptor"  # Assuming this is the target protein
    actual_binder_col = "peptide"   # Assuming this is the binder

    try:
        ds = ds.rename_columns({actual_target_col: "target", actual_binder_col: "binder"})
        ds = ds.remove_columns([col for col in ds.column_names if col not in {"target", "binder"}])
    except ValueError as e:
        logger.error(f"Error renaming columns: {e}")
        logger.error(f"Available columns: {ds.column_names}")
        # If we can't rename, try to select columns directly
        if actual_target_col in ds.column_names and actual_binder_col in ds.column_names:
            ds = ds.select_columns([actual_target_col, actual_binder_col])
            ds = ds.rename_columns({actual_target_col: "target", actual_binder_col: "binder"})
        else:
            # If we still can't get the right columns, create a simple dataset with dummy data
            # This is just to allow testing the environment without the actual dataset
            logger.warning("Using dummy data since the dataset columns don't match the expected format!")
            dummy_data = {
                "target": ["MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEY"] * 10,
                "binder": ["PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASKA"] * 10
            }
            ds = Dataset.from_dict(dummy_data)

    return ds

def get_pdb_chain_details(pdb_content: str, preview_lines: int = 10) -> Tuple[Dict[str, Dict[str, int]], str]:
    """
    Parses PDB content to extract detailed information for each chain.

    Returns:
        A tuple containing:
        - chain_details (Dict[str, Dict[str, int]]):
            A dictionary where keys are chain IDs (e.g., "A").
            Each value is another dictionary:
            {
                "min_residue": int,  # Smallest residue number found for this chain
                "max_residue": int,  # Largest residue number found for this chain
                "length": int       # Count of unique C-alpha atoms (residues) in this chain
            }
        - pdb_preview (str): A string preview of the PDB content.
    """
    chain_info_temp: Dict[str, Dict[str, Union[Set[int], int]]] = {} # Stores residue numbers and CA count for each chain
    atom_lines = []
    header_lines = []

    # First pass: Collect all residue numbers and CA atoms per chain
    for line in pdb_content.splitlines():
        if line.startswith("ATOM"): # Consider only ATOM records for canonical residues
            atom_lines.append(line)
            chain_id = line[21:22].strip()
            if not chain_id:
                chain_id = " " # Default for blank chain ID, consider how RFDiffusion handles this

            atom_name = line[12:16].strip()

            try:
                residue_num = int(line[22:26].strip())

                if chain_id not in chain_info_temp:
                    chain_info_temp[chain_id] = {"residues": set(), "ca_count": 0}

                chain_info_temp[chain_id]["residues"].add(residue_num)
                if atom_name == "CA":
                    chain_info_temp[chain_id]["ca_count"] += 1
            except ValueError:
                logger.warning(f"Could not parse residue number from PDB line: {line}")
                continue
        elif line.startswith("HEADER") or line.startswith("TITLE") or line.startswith("COMPND"):
            header_lines.append(line)

    # Second pass: Calculate min, max, and length from collected data
    chain_details: Dict[str, Dict[str, int]] = {}
    for chain_id, data in chain_info_temp.items():
        if data["residues"]: # Only process if residues were found
            min_res = min(data["residues"])
            max_res = max(data["residues"])
            # Length can be defined in two ways:
            # 1. max_res - min_res + 1 (if contiguous numbering)
            # 2. Count of unique residues (safer for gaps, but AF2 is usually contiguous)
            # 3. Count of C-alpha atoms (good proxy for actual modeled residues)
            # Let's use ca_count as it reflects actual modeled residues.
            # If ca_count is 0 but residues were found (e.g. only HETATMs), this needs thought.
            # For now, prioritizing ca_count.
            length = data["ca_count"] if data["ca_count"] > 0 else len(data["residues"])

            chain_details[chain_id] = {
                "min_residue": min_res,
                "max_residue": max_res,
                "length": length
            }
        else:
            logger.warning(f"Chain {chain_id} had no parseable ATOM residue numbers.")


    # Construct PDB preview
    preview_str_parts = header_lines[:min(len(header_lines), preview_lines // 2)]
    remaining_preview_lines = preview_lines - len(preview_str_parts)
    preview_str_parts.extend(atom_lines[:min(len(atom_lines), remaining_preview_lines)])
    pdb_preview = "\n".join(preview_str_parts)
    if len(pdb_content.splitlines()) > preview_lines:
        pdb_preview += "\n..."

    return chain_details, pdb_preview

def get_pdb_chain_lengths_and_preview(pdb_content: str, preview_lines: int = 10) -> Tuple[Dict[str, int], str]:
    chain_lengths = {}
    current_chain_id = None
    max_residue_num = 0
    atom_lines = []
    header_lines = []

    for line in pdb_content.splitlines():
        if line.startswith("ATOM") or line.startswith("HETATM"):
            atom_lines.append(line)
            chain_id = line[21:22].strip()
            if not chain_id: # Handle cases where chain ID might be blank
                chain_id = " " # Default to space if blank, or handle as error
            try:
                residue_num = int(line[22:26].strip())
                if current_chain_id != chain_id:
                    if current_chain_id is not None: # For previous chain
                        chain_lengths[current_chain_id] = max_residue_num
                    current_chain_id = chain_id
                    max_residue_num = residue_num # Reset for new chain
                else:
                    max_residue_num = max(max_residue_num, residue_num)
            except ValueError:
                continue # Skip if residue number is not an int
        elif line.startswith("HEADER") or line.startswith("TITLE") or line.startswith("COMPND"):
            header_lines.append(line)

    if current_chain_id is not None: # Store the last chain's length
        chain_lengths[current_chain_id] = max_residue_num

    preview_str_parts = header_lines[:min(len(header_lines), preview_lines // 2)]
    remaining_preview_lines = preview_lines - len(preview_str_parts)
    preview_str_parts.extend(atom_lines[:min(len(atom_lines), remaining_preview_lines)])

    pdb_preview = "\n".join(preview_str_parts)
    if len(pdb_content.splitlines()) > preview_lines:
        pdb_preview += "\n..."

    return chain_lengths, pdb_preview

def construct_user_prompt(state: dict) -> str: # state is an item from self.episodes_state
    internal_step = state.get("current_internal_step", 0)
    target_sequence = state.get("target_sequence")
    user_prompt_str = ""

    if internal_step == 0: # Step 1: Predict Target Structure (AlphaFold2)
        user_prompt_str = (
            f"The target protein sequence is: {target_sequence}. "
            "Your first task is to predict its 3D structure using the 'predict_target_structure_alphafold2' tool. "
            "You must provide the 'sequence' argument."
        )
    elif internal_step == 1: # Step 2: Design Binder Backbone (RFDiffusion)
        target_pdb_preview = state.get("target_pdb_preview", "PDB preview not available.") # Can keep preview for general context

        # --- NEW CHAIN INFO FORMATTING ---
        chain_details = state.get("target_chain_details", {}) # Get the new detailed info
        if chain_details:
            chain_info_parts = []
            for chain_id, details in chain_details.items():
                min_r = details.get('min_residue', 'N/A')
                max_r = details.get('max_residue', 'N/A')
                l = details.get('length', 'N/A')
                chain_info_parts.append(f"Chain {chain_id} (Residues: {min_r}-{max_r}, Length: {l} amino acids)")
            chain_info_str = "\n- ".join(chain_info_parts)
            if chain_info_str:
                 chain_info_str = "- " + chain_info_str # Add leading bullet for the first item
        else:
            chain_info_str = "Chain information not available or PDB not yet processed."
        # --- END NEW CHAIN INFO FORMATTING ---

        user_prompt_str = (
            f"The 3D structure of the target protein has been predicted.\n"
            # Optional: f"Target PDB preview:\n{target_pdb_preview}\n\n"
            f"Target Protein Chain Details:\n{chain_info_str}\n\n" # Use the detailed chain info
            "Your task is to design a binder backbone using the 'design_binder_backbone_rfdiffusion' tool. "
            "You MUST specify 'contigs' for this tool. The 'contigs' string defines segments from the target PDB and segments for the new binder. "
            "Examples:\n"
            "  - To use residues 10 through 100 of target chain A, and then diffuse a 60-residue binder: 'A10-100/0 60'\n"
            "  - To use chain B from residue 5 to 50, then diffuse a 30-residue binder, then use chain B from residue 60 to 100: 'B5-50/0 30 B60-100'\n"
            "You MUST use the chain IDs and residue ranges exactly as provided in the 'Target Protein Chain Details' above. "
            "Do not invent chains or residue numbers outside these specified ranges for the target segments. "
            "For binder segments (e.g., '/0 60'), specify the desired length (e.g., 60).\n"
            "Optionally, provide 'hotspot_residues' (e.g., ['A50', 'A52']), ensuring these residues exist on the target as per the details above."
        )
    elif internal_step == 2: # Step 3: Design Binder Sequence (ProteinMPNN)
        # Get detailed binder chain information using the get_pdb_chain_details function
        binder_pdb_content = state.get("binder_backbone_pdb_content")
        if binder_pdb_content:
            binder_chain_details, binder_pdb_preview = get_pdb_chain_details(binder_pdb_content)
            binder_chain_info_str = "\n- ".join([f"Chain {cID} (Residues: {d.get('min_residue','N/A')}-{d.get('max_residue','N/A')}, Length: {d.get('length','N/A')})" for cID, d in binder_chain_details.items()])
            if binder_chain_info_str: binder_chain_info_str = "- " + binder_chain_info_str
        else:
            binder_pdb_preview = "Binder PDB preview not available."
            binder_chain_info_str = "Binder chain information not available."

        user_prompt_str = (
            f"A binder backbone has been generated. Binder PDB preview:\n{binder_pdb_preview}\n"
            f"Binder chain information:\n{binder_chain_info_str}.\n"
            "Now, design an optimal amino acid sequence for this binder backbone using the 'design_binder_sequence_proteinmpnn' tool. "
            "You can optionally specify 'sampling_temp' (e.g., [0.1, 0.2])."
        )
    elif internal_step == 3: # Step 4: Evaluate Complex (AlphaFold2-Multimer)
        designed_binder_seq_data = state.get("designed_binder_sequence") # This is List[str]

        binder_display_str = "Not available"
        if isinstance(designed_binder_seq_data, list) and designed_binder_seq_data:
            if len(designed_binder_seq_data) == 1:
                binder_display_str = designed_binder_seq_data[0]
            else:
                binder_display_str = f"{len(designed_binder_seq_data)} chains: " + \
                                     ", ".join([f"Chain {i+1} ({len(s)} aa): {s[:20]}..."
                                                for i, s in enumerate(designed_binder_seq_data)])
        elif isinstance(designed_binder_seq_data, str): # Should not happen with new PMPNN parsing
             binder_display_str = designed_binder_seq_data

        user_prompt_str = (
            f"A binder has been designed. Designed binder sequence(s): {binder_display_str}. "
            f"The original target sequence was: {target_sequence[:60]}...\n"
            "Finally, evaluate the binding complex of the original target protein and ALL chains of this designed binder using the "
            "'evaluate_binder_complex_alphafold2_multimer' tool. "
            "You can optionally specify 'relax_prediction' (default is True)."
        )
    else: # Workflow complete or error
        user_prompt_str = "The protein design workflow is complete. No further actions required by you for this item. If successful, the key metric was the pLDDT of the complex."

    # Retry logic should remain the same:
    if state.get("retry_count_this_internal_step", 0) > 0 and internal_step < 4:
        retry_prefix = "Your previous attempt at this step was not successful. "
        if state.get("previous_tool_error_message"):
            retry_prefix += f"Details: {state['previous_tool_error_message']}. "
        retry_prefix += "Please review the requirements and PDB details carefully and try again to correctly use the expected tool.\n\n"
        user_prompt_str = retry_prefix + user_prompt_str

    return user_prompt_str


class BinderRow(TypedDict):
    target: str
    binder: str


# Define a configuration class for BinderBenchEnv
class BinderBenchConfig(BaseEnvConfig):
    nim_api_key: Optional[str] = Field(None, description="NVIDIA NIM API key")
    nim_api_base_url: str = Field("https://health.api.nvidia.com/v1", description="NIM API base URL")
    api_timeout: int = Field(1800, description="Timeout for NIM API calls")  # Increased default
    polling_interval: int = Field(30, description="Polling interval for NIM jobs")  # Increased default
    output_dir: str = Field(default=str(Path(__file__).parent / "outputs"), description="Directory to save PDBs, etc.")
    debug_protein_design_calls: bool = Field(False, description="Enable debug mode for NIM protein API calls, returning mock data.")
    max_retries_per_internal_step: int = Field(100, description="Max retries for a failed tool call within a workflow step (0 means no retries).")  # Default to 1 retry (2 attempts total)
    # Dataset specific
    dataset_name: str = Field("ronig/protein_binding_sequences", description="Dataset for target sequences")
    target_col: str = Field("receptor", description="Target column name (actual column in the dataset)")
    binder_col: str = Field("peptide", description="Binder column name (actual column in the dataset)")
    # Scoring weights
    metric_weights: Dict[str, float] = Field(
        default={"plddt": 0.3, "ptm": 0.3, "iptm": 0.4},
        description="Weights for combining scoring metrics for complex_quality"
    )


class BinderBenchEnv(BaseEnv):
    name = "binderbench"
    env_config_cls = BinderBenchConfig  # Use the new config class

    def __init__(self, config: BinderBenchConfig, server_configs: List[APIServerConfig], slurm=False, testing=False):
        super().__init__(config, server_configs, slurm, testing)
        self.config: BinderBenchConfig  # Type hint for convenience

        # Initialize with process_mode=False (will be set to True when running with process command)
        self.process_mode = False

        # Tool definitions for LLM function calling
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "predict_target_structure_alphafold2",  # Renamed for clarity
                    "description": "Predicts the 3D structure of the target protein sequence using AlphaFold2.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sequence": {"type": "string", "description": "Amino acid sequence of the target protein."},
                        },
                        "required": ["sequence"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "design_binder_backbone_rfdiffusion",
                    "description": "Generates a novel protein binder backbone using RFDiffusion, conditioned on the target protein structure.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            # target_pdb_content will be implicitly taken from state
                            "contigs": {"type": "string", "description": "RFDiffusion contigs (e.g., 'A1-100/0 50-70')."},
                            "hotspot_residues": {"type": "array", "items": {"type": "string"}, "description": "Optional hotspot residues (e.g., ['A50', 'A52'])."},
                        },
                        "required": ["contigs"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "design_binder_sequence_proteinmpnn",
                    "description": "Designs an amino acid sequence for the generated binder backbone.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            # binder_backbone_pdb_content taken from state
                            "sampling_temp": {"type": "array", "items": {"type": "number"}, "description": "Sampling temperatures (e.g., [0.1, 0.2]). Default [0.1]."}
                        },
                        "required": []  # sampling_temp is optional
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "evaluate_binder_complex_alphafold2_multimer",
                    "description": "Predicts the complex structure of target and designed binder, providing quality scores.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            # target_sequence and binder_sequence taken from state
                            "relax_prediction": {"type": "boolean", "description": "Whether to relax the prediction. Default True."}
                        },
                        "required": []  # relax_prediction is optional
                    }
                }
            }
        ]

        # Ensure output directory exists
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.episodes_state = {}  # To store state for each item_id
        self._debug_af2m_call_count = 0  # For debug mode pLDDT alternation

        self.completed_episode_metrics: List[Dict] = []  # Store completed workflow metrics for evaluation
        self.rollouts_for_wandb = []  # Initialize buffer for WandB rollout data

    async def _execute_tool(self, tool_name: str, args: Dict, workflow_state: Dict) -> Dict:
        """Executes the specified NIM tool and updates the workflow_state."""
        item_id = workflow_state["item_id"]
        internal_step = workflow_state["current_internal_step"]
        logger.info(f"Workflow {item_id}, Internal Step {internal_step}: Executing tool '{tool_name}' with args: {args}")

        # Ensure API key is available
        if not self.config.nim_api_key:
            logger.error(f"NIM API key not configured for tool {tool_name}.")
            return {"success": False, "error": "NIM API key not configured."}

        result = {"success": False, "error": "Unknown tool or execution error."}
        try:
            if tool_name == "predict_target_structure_alphafold2":
                result = await self._run_nim_alphafold2(args, workflow_state)
            elif tool_name == "design_binder_backbone_rfdiffusion":
                result = await self._run_nim_rfdiffusion(args, workflow_state)
            elif tool_name == "design_binder_sequence_proteinmpnn":
                result = await self._run_nim_proteinmpnn(args, workflow_state)
            elif tool_name == "evaluate_binder_complex_alphafold2_multimer":
                result = await self._run_nim_af2_multimer(args, workflow_state)
            else:
                result = {"success": False, "error": f"Unknown tool name: {tool_name}"}
        except Exception as e:
            logger.error(f"Workflow {item_id}, Step {internal_step}: Exception during tool '{tool_name}': {e}", exc_info=True)
            result = {"success": False, "error": str(e)}

        # The runner methods should have updated workflow_state directly
        return result

    async def _run_nim_alphafold2(self, args: Dict, workflow_state: Dict) -> Dict:
        item_id = workflow_state["item_id"] # Get item_id for logging and unique filenames

        # ***** START DEBUG MODE LOGIC FOR ALPHAFOLD2 *****
        if self.config.debug_protein_design_calls:
            logger.warning(f"DEBUG MODE: Bypassing AlphaFold2 API call for workflow {item_id}.")

            # Define the path to your fixed PDB file - use absolute path in the project root
            # Create a Path object for the PDB file in the project root
            project_root = Path(__file__).resolve().parent.parent.parent.parent  # One more level up to reach atropos root
            fixed_pdb_path = project_root / "binder_outputs" / "target.pdb"

            if not fixed_pdb_path.exists():
                logger.error(f"DEBUG MODE ERROR: Fixed PDB file not found at {fixed_pdb_path}. Cannot proceed with mock AF2 output.")
                # Create a dummy PDB content if file not found to prevent downstream errors, but log severe warning
                pdb_content = "HEADER DUMMY PDB FOR DEBUG - TARGET.PDB NOT FOUND\nATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\nTER\nEND\n"
                workflow_state["target_pdb_content"] = pdb_content
                chain_details, pdb_preview = get_pdb_chain_details(pdb_content) # Use the new function
                workflow_state["target_chain_details"] = chain_details # Store detailed info
                workflow_state["target_pdb_preview"] = pdb_preview
                workflow_state["target_structure_predicted"] = True # Mark as "predicted" for workflow to proceed
                return {"success": False, "error": f"Debug mode error: {fixed_pdb_path} not found.", "target_pdb_preview": pdb_preview}

            try:
                with open(fixed_pdb_path, "r") as f:
                    pdb_content = f.read()

                workflow_state["target_pdb_content"] = pdb_content
                chain_details, pdb_preview = get_pdb_chain_details(pdb_content) # Use the new function
                workflow_state["target_chain_details"] = chain_details # Store detailed info
                workflow_state["target_pdb_preview"] = pdb_preview
                workflow_state["target_structure_predicted"] = True

                # Optionally, save a copy of this mock PDB to the usual output location for consistency
                debug_output_pdb_path = self.output_dir / f"target_{item_id}_s{workflow_state['current_internal_step']}_af2_DEBUG.pdb"
                with open(debug_output_pdb_path, "w") as f: f.write(pdb_content)
                logger.info(f"DEBUG MODE: Used fixed PDB from {fixed_pdb_path}. Copied to {debug_output_pdb_path}. Chain details: {chain_details}")

                return {"success": True, "message": "DEBUG MODE: Used fixed PDB for AlphaFold2.", "target_pdb_preview": pdb_preview}
            except Exception as e:
                logger.error(f"DEBUG MODE ERROR: Failed to read or process {fixed_pdb_path}: {e}", exc_info=True)
                return {"success": False, "error": f"Debug mode error: Failed processing {fixed_pdb_path}."}
        # ***** END DEBUG MODE LOGIC FOR ALPHAFOLD2 *****

        # --- Original API call logic ---
        sequence = args.get("sequence")
        if not sequence:
            return {"success": False, "error": "Missing 'sequence' for AlphaFold2."}
        # Ensuring LLM uses the canonical target sequence from state
        if sequence != workflow_state["target_sequence"]:
             logger.warning(f"LLM provided sequence '{sequence[:20]}...' for AF2, but expected target '{workflow_state['target_sequence'][:20]}...'. Using expected target from workflow state.")
             sequence = workflow_state["target_sequence"]

        # (Rest of your existing _run_nim_alphafold2 logic for actual API call...)
        api_result = await call_alphafold2(
            sequence=sequence, api_key=self.config.nim_api_key,
            timeout=self.config.api_timeout, polling_interval=self.config.polling_interval
        )
        if api_result and isinstance(api_result, list) and api_result[0]:
            pdb_content = api_result[0]
            workflow_state["target_pdb_content"] = pdb_content
            chain_details, pdb_preview = get_pdb_chain_details(pdb_content) # Use the new function
            workflow_state["target_chain_details"] = chain_details # Store detailed info
            workflow_state["target_pdb_preview"] = pdb_preview
            workflow_state["target_structure_predicted"] = True
            pdb_path = self.output_dir / f"target_{item_id}_s{workflow_state['current_internal_step']}_af2.pdb"
            with open(pdb_path, "w") as f: f.write(pdb_content)
            logger.info(f"Workflow {item_id}: AlphaFold2 PDB saved to {pdb_path}. Chain details: {chain_details}")
            return {"success": True, "message": "AlphaFold2 prediction complete.", "target_pdb_preview": pdb_preview}
        else:
            logger.error(f"Workflow {item_id}: AlphaFold2 call failed or returned unexpected data: {api_result}")
            return {"success": False, "error": "AlphaFold2 prediction failed."}

    async def _run_nim_rfdiffusion(self, args: Dict, workflow_state: Dict) -> Dict:
        target_pdb_content = workflow_state.get("target_pdb_content")
        contigs = args.get("contigs")
        if not target_pdb_content: return {"success": False, "error": "Target PDB not found in state for RFDiffusion."}
        if not contigs: return {"success": False, "error": "Missing 'contigs' for RFDiffusion."}

        hotspot_residues = args.get("hotspot_residues") # Optional
        item_id = workflow_state["item_id"]
        api_result = await call_rfdiffusion(
            input_pdb=target_pdb_content, api_key=self.config.nim_api_key,
            contigs=contigs, hotspot_res=hotspot_residues,
            timeout=self.config.api_timeout, polling_interval=self.config.polling_interval
            # Add other RFD specific params
        )
        if api_result and "output_pdb" in api_result:
            binder_pdb = api_result["output_pdb"]
            workflow_state["binder_backbone_pdb_content"] = binder_pdb
            workflow_state["binder_backbone_designed"] = True
            pdb_path = self.output_dir / f"binder_backbone_{item_id}_s{workflow_state['current_internal_step']}_rfd.pdb"
            with open(pdb_path, "w") as f: f.write(binder_pdb)
            logger.info(f"Workflow {item_id}: RFDiffusion PDB saved to {pdb_path}")
            # NO LONGER INCREMENT current_internal_step HERE - collect_trajectories will handle this
            return {"success": True, "message": "RFDiffusion complete.", "binder_backbone_pdb_preview": binder_pdb[:150] + "..."}
        else:
            logger.error(f"Workflow {item_id}: RFDiffusion call failed or returned unexpected data: {api_result}")
            return {"success": False, "error": "RFDiffusion failed."}

    async def _run_nim_proteinmpnn(self, args: Dict, workflow_state: Dict) -> Dict:
        binder_pdb = workflow_state.get("binder_backbone_pdb_content")
        if not binder_pdb:
            return {"success": False, "error": "Binder backbone PDB not found for ProteinMPNN."}

        sampling_temp_list = args.get("sampling_temp", [0.1])
        item_id = workflow_state["item_id"]

        api_result = await call_proteinmpnn(
            input_pdb=binder_pdb, api_key=self.config.nim_api_key,
            sampling_temp=sampling_temp_list,
            timeout=self.config.api_timeout, polling_interval=self.config.polling_interval
        )

        if not (api_result and "mfasta" in api_result):
            logger.error(f"Workflow {item_id}: ProteinMPNN call failed or returned unexpected data: {api_result}")
            return {"success": False, "error": "ProteinMPNN call failed or no mfasta in result."}

        fasta_content = api_result["mfasta"]
        logger.info(f"CRITICAL_DEBUG: ProteinMPNN raw mfasta output for item {item_id}:\n{fasta_content}")

        # --- FASTA Parsing Logic to find best sequence by global_score ---
        entries: List[Tuple[float, str, str]] = []  # (global_score, header, sequence_line)
        current_header = None
        current_sequence_parts: List[str] = []

        for line_content in fasta_content.splitlines():
            line = line_content.strip()
            if not line: continue

            if line.startswith(">"):
                if current_header and current_sequence_parts: # Process previous entry
                    full_sequence_line = "".join(current_sequence_parts)
                    score_match = re.search(r"global_score=([-\d.]+)", current_header)
                    global_score = float(score_match.group(1)) if score_match else -float('inf')
                    entries.append((global_score, current_header, full_sequence_line))
                current_header = line
                current_sequence_parts = []
            else:
                current_sequence_parts.append(line)

        if current_header and current_sequence_parts: # Process the last entry
            full_sequence_line = "".join(current_sequence_parts)
            score_match = re.search(r"global_score=([-\d.]+)", current_header)
            global_score = float(score_match.group(1)) if score_match else -float('inf')
            entries.append((global_score, current_header, full_sequence_line))

        if not entries:
            logger.error(f"Workflow {item_id}: No sequences found in ProteinMPNN mfasta output.")
            return {"success": False, "error": "No sequences parsed from ProteinMPNN mfasta."}

        # Sort by global_score (descending) and select the best
        entries.sort(key=lambda x: x[0], reverse=True)
        best_global_score, best_header, best_full_sequence_line = entries[0]

        logger.info(f"Workflow {item_id}: Best PMPNN sequence chosen (global_score={best_global_score:.4f}) from header: '{best_header}'")
        logger.info(f"Workflow {item_id}: Corresponding sequence line: '{best_full_sequence_line}'")

        # Split the selected sequence line by '/' to handle potential chainbreaks
        parsed_binder_chains: List[str] = [
            seq_part.strip() for seq_part in best_full_sequence_line.split('/') if seq_part.strip()
        ]

        if not parsed_binder_chains:
            error_msg = f"Splitting best PMPNN sequence ('{best_full_sequence_line}') by '/' yielded no valid chains."
            logger.error(f"Workflow {item_id}: {error_msg}")
            return {"success": False, "error": error_msg}

        # Validate each parsed chain (ensure they are valid protein sequences)
        for seq_idx, seq_part in enumerate(parsed_binder_chains):
            if not (seq_part and seq_part.isalpha() and seq_part.isupper()):
                error_msg = f"Parsed binder chain {seq_idx+1} ('{seq_part[:30]}...') contains invalid characters or is empty."
                logger.error(f"Workflow {item_id}: {error_msg}")
                return {"success": False, "error": error_msg}

        workflow_state["designed_binder_sequence"] = parsed_binder_chains # Store as List[str]
        workflow_state["binder_sequence_designed"] = True

        fasta_path = self.output_dir / f"binder_sequence_{item_id}_s{workflow_state['current_internal_step']}_pmpnn.fasta"
        with open(fasta_path, "w") as f: f.write(fasta_content) # Save original full FASTA
        logger.info(f"Workflow {item_id}: ProteinMPNN FASTA saved to {fasta_path}. Selected binder chains: {parsed_binder_chains}")

        preview = parsed_binder_chains[0][:60] + "..." if len(parsed_binder_chains[0]) > 60 else parsed_binder_chains[0]
        if len(parsed_binder_chains) > 1:
            preview += f" (+ {len(parsed_binder_chains)-1} more chain(s))"

        return {
            "success": True,
            "message": f"ProteinMPNN complete. Selected best (global_score={best_global_score:.4f}).",
            "designed_binder_sequence_list": parsed_binder_chains,
            "designed_binder_sequence_preview": preview
        }

    async def _run_nim_af2_multimer(self, args: Dict, workflow_state: Dict) -> Dict:
        target_seq = workflow_state.get("target_sequence")
        binder_seq_data = workflow_state.get("designed_binder_sequence")

        if not target_seq:
            return {"success": False, "error": "Missing target sequence for AlphaFold2-Multimer."}

        if not binder_seq_data:
            return {"success": False, "error": "Missing binder sequence for AlphaFold2-Multimer."}

        # Handle binder_seq_data which could now be either a List[str] or a single string (for backward compatibility)
        binder_sequences = []
        if isinstance(binder_seq_data, list):
            binder_sequences = binder_seq_data
        elif isinstance(binder_seq_data, str):
            binder_sequences = [binder_seq_data]  # Wrap in list
        else:
            return {"success": False, "error": f"Unexpected type for binder sequence: {type(binder_seq_data)}"}

        if not binder_sequences:
            return {"success": False, "error": "Empty binder sequence list for AlphaFold2-Multimer."}

        relax = args.get("relax_prediction", True)
        item_id = workflow_state["item_id"]

        # Log all sequences for debugging
        total_binder_length = sum(len(seq) for seq in binder_sequences)
        logger.info(f"Workflow {item_id}: Running AlphaFold2-Multimer with target (len {len(target_seq)}) and {len(binder_sequences)} binder chain(s) (total len {total_binder_length}).")

        # Check if in debug mode
        if self.config.debug_protein_design_calls:
            # Increment the counter for alternating results
            self._debug_af2m_call_count += 1
            logger.warning(f"DEBUG MODE: Using mock data for AlphaFold2-Multimer (call #{self._debug_af2m_call_count})")

            # Create mock results that alternate between high and low quality scores
            # For odd-numbered calls (1, 3, 5...) - return high quality
            # For even-numbered calls (2, 4, 6...) - return low quality
            if self._debug_af2m_call_count % 2 == 1:  # Odd calls
                mock_plddt = 87.5  # Good score
                success_message = "DEBUG MODE: Returning high-quality mock results"
            else:  # Even calls
                mock_plddt = 45.2  # Poor score
                success_message = "DEBUG MODE: Returning low-quality mock results"

            # Create a mock PDB file path
            mock_pdb_path = self.output_dir / f"mock_complex_{item_id}_af2m.pdb"
            with open(mock_pdb_path, "w") as f:
                f.write(f"MOCK PDB FILE with pLDDT {mock_plddt}\nFor debug purposes only.\n")

            # Update workflow state with the mock values
            workflow_state["complex_pdb_content_path"] = str(mock_pdb_path)
            workflow_state["af2_multimer_plddt"] = mock_plddt
            workflow_state["af2_multimer_ptm"] = 0.0
            workflow_state["af2_multimer_iptm"] = 0.0
            workflow_state["complex_evaluated"] = True

            logger.info(f"Workflow {item_id}: {success_message}. Mock pLDDT: {mock_plddt:.2f}")
            return {
                "success": True,
                "message": f"{success_message}. Mock pLDDT: {mock_plddt:.2f}",
                "plddt": mock_plddt,
                "ptm": 0.0,
                "iptm": 0.0,
                "complex_file_path": str(mock_pdb_path)
            }

        # Non-debug mode: proceed with actual API call
        # Create a list with target sequence as first element, followed by all binder sequences
        all_sequences = [target_seq] + binder_sequences

        logger.info(f"Workflow {item_id}: Calling AlphaFold2-Multimer with {len(all_sequences)} sequences: "
                   f"1 target ({len(target_seq)} aa) + {len(binder_sequences)} binder chains.")

        api_result = await call_alphafold2_multimer(
            sequences=all_sequences,  # Pass all sequences as a flat list: [target_seq, binder_seq1, binder_seq2, ...]
            api_key=self.config.nim_api_key,
            relax_prediction=relax,
            timeout=self.config.api_timeout,
            polling_interval=self.config.polling_interval
        )

        if api_result and api_result.get("structures") and len(api_result["structures"]) > 0:
            # Assuming the first structure in the list is the one we care about (e.g., rank 1 model)
            # Or if NIM usually returns only one model's PDB in the .response
            primary_structure_info = api_result["structures"][0]

            plddt = primary_structure_info.get("average_plddt", 0.0)
            # ipTM and pTM are explicitly set to None or not parsed by _process_nvidia_zip_output
            # So, we don't expect them here unless _process_nvidia_zip_output is changed to parse them.
            # For now, focus on pLDDT.

            workflow_state["complex_pdb_content_path"] = primary_structure_info.get("saved_pdb_path") # Path to the extracted PDB
            workflow_state["af2_multimer_plddt"] = plddt
            workflow_state["af2_multimer_ptm"] = 0.0 # Explicitly 0 or None if not calculating
            workflow_state["af2_multimer_iptm"] = 0.0 # Explicitly 0 or None
            workflow_state["complex_evaluated"] = True

            logger.info(f"Workflow {item_id}: AlphaFold2-Multimer complete. Average pLDDT: {plddt:.2f}")
            return {
                "success": True,
                "message": f"AlphaFold2-Multimer evaluation complete. Average pLDDT: {plddt:.2f}.",
                "plddt": plddt,
                "ptm": 0.0, # Reflect that we are not using it from here
                "iptm": 0.0, # Reflect that we are not using it from here
                "complex_file_path": str(primary_structure_info.get("saved_pdb_path"))
            }
        else:
            error_msg = "AlphaFold2-Multimer call failed or did not return expected structure data."
            if api_result and "error" in api_result: # If call_alphafold2_multimer returned an error dict
                error_msg = api_result["error"]
            logger.error(f"Workflow {item_id}: {error_msg}. API result: {api_result}")
            workflow_state["complex_evaluated"] = False # Mark as not evaluated
            return {"success": False, "error": error_msg}

    @classmethod
    def config_init(cls) -> Tuple[BinderBenchConfig, List[APIServerConfig]]:
        # Set defaults from config or environment variables

        # Load from a potential binderbench_default.yaml file if it exists
        default_yaml_path = Path(__file__).parent / "configs" / "binderbench_default.yaml"
        yaml_config_values = {}
        if default_yaml_path.exists():
            with open(default_yaml_path, 'r') as f:
                yaml_config_values = yaml.safe_load(f) or {}

        # Create environment config with priority order:
        # 1. Environment variables (e.g., NIM API key)
        # 2. YAML config values
        # 3. Default values from Field definitions
        env_config = BinderBenchConfig(
            # --- BaseEnvConfig fields relevant to WandB ---
            use_wandb=True,  # Enable WandB by default
            wandb_name=cls.name,  # Uses BinderBenchEnv.name as the base for run names
            # num_rollouts_to_keep and num_rollouts_per_group_for_logging are already in BaseEnvConfig
            # include_messages will be True by default for process_mode, can be overridden for serve

            # --- BinderBenchConfig specific fields ---
            nim_api_key=os.environ.get("NVIDIA_NIM_API_KEY"),
            debug_protein_design_calls=yaml_config_values.get(
                "debug_protein_design_calls",
                bool(os.environ.get("DEBUG_PROTEIN_DESIGN_CALLS", False))
            ),
            # Other config properties use defaults from Field definitions
        )

        # Setup default server configs
        llm_api_key = os.environ.get("OPENAI_API_KEY")
        llm_base_url = os.environ.get("OPENAI_API_BASE")

        server_configs = [
            APIServerConfig(
                model_name=os.environ.get("DEFAULT_LLM_MODEL", "gpt-4-turbo"),
                api_key=llm_api_key,
                base_url=llm_base_url  # Will be None if OPENAI_API_BASE not set
            )
        ]
        return env_config, server_configs

    async def setup(self):
        self.iter = 0
        self.train = load_target_binder_pairs(
                dataset_name=self.config.dataset_name, # Use config
                target_col=self.config.target_col,     # Use config
                binder_col=self.config.binder_col      # Use config
            )
        # self.train.shuffle() # Shuffle is good, but might make iter less predictable for debugging
        logger.info(f"Loaded {len(self.train)} target-binder pairs for {self.name}.")

        # Validate API key
        if not self.config.nim_api_key:
            self.config.nim_api_key = os.environ.get("NVIDIA_NIM_API_KEY")
            if not self.config.nim_api_key:
                logger.warning("NVIDIA NIM API key not set. Protein design functions may not work properly.")

    def _initialize_workflow_state(self, item_id: str, target_sequence: str, ground_truth_binder: Optional[str]) -> Dict:
        """Initializes or resets the state for a new workflow."""
        return {
            "item_id": item_id,
            "current_internal_step": 0,
            "target_sequence": target_sequence,
            "ground_truth_binder_sequence": ground_truth_binder, # Store for final evaluation
            "target_pdb_content": None,
            "target_chain_details": None, # Store detailed chain information
            "binder_backbone_pdb_content": None,
            "designed_binder_sequence": None,
            "complex_pdb_content_path": None, # Path to AF2-Multimer output
            "af2_multimer_plddt": 0.0,
            "af2_multimer_ptm": 0.0,
            "af2_multimer_iptm": 0.0,
            "target_structure_predicted": False,
            "binder_backbone_designed": False,
            "binder_sequence_designed": False,
            "complex_evaluated": False,
            "workflow_complete_flag": False, # Flag to mark end of workflow
            "last_tool_success": True, # Track if the last tool call was successful
            "cumulative_reward": 0.0, # For multi-step reward accumulation
            "turn_messages_history": [], # Store list of (List[Message]) for each turn
            "retry_count_this_internal_step": 0, # ***** ADDED: Tracks retries for the current internal_step *****
            "previous_tool_error_message": None, # ***** ADDED: To inform LLM on retry *****
        }

    async def get_next_item(self) -> Item:
        """
        Provides the initial information for a new protein design workflow.
        Returns an Item tuple: (item_id, initial_target_sequence_info)
        """
        raw_item: BinderRow = self.train[self.iter % len(self.train)]
        self.iter += 1

        item_id = str(uuid.uuid4())
        target_sequence = raw_item["target"]
        ground_truth_binder = raw_item.get("binder") # May not always be used for de novo

        # Store the initial state for this new workflow
        self.episodes_state[item_id] = self._initialize_workflow_state(item_id, target_sequence, ground_truth_binder)

        # The "item" for Atropos's collect_trajectories will just be the item_id.
        # The actual data is pulled from self.episodes_state[item_id].
        return item_id # Item is now just the ID. Initial step is always 0 for a new workflow.

    # reset_state is effectively handled by _initialize_workflow_state and get_next_item
    def reset_state(self, item_id: str) -> dict:
        """Retrieves the workflow state for the given item_id."""
        if item_id in self.episodes_state:
            return self.episodes_state[item_id]
        else:
            # This should ideally never happen
            logger.error(f"No state found for item_id {item_id}. Creating a default state.")
            return self._initialize_workflow_state(item_id, "", None)  # Empty default state

    async def collect_trajectories(self, item_id: str) -> Tuple[Optional[ScoredDataGroup], List[Item]]:
        workflow_state = self.episodes_state.get(item_id)
        if not workflow_state:
            logger.error(f"Workflow state for item_id {item_id} not found. Skipping.")
            return None, []

        if workflow_state.get("workflow_complete_flag"):
            logger.info(f"Workflow for {item_id} already marked complete. Skipping.")
            # Optionally, clean up here if you don't want to re-process completed items
            # if item_id in self.episodes_state: del self.episodes_state[item_id]
            return None, []

        is_processing_mode = getattr(self, 'process_mode', False) # Check the flag

        if is_processing_mode:
            # --- PROCESS MODE: Run full workflow, aggregate all turns ---
            all_turns_data_for_jsonl = [] # To store data for each turn for one JSONL line
            MAX_INTERNAL_STEPS = 4 # AF2, RFD, PMPNN, AF2M

            while workflow_state["current_internal_step"] < MAX_INTERNAL_STEPS and \
                  not workflow_state.get("workflow_complete_flag"):

                # Construct prompt (will include retry info if applicable)
                current_turn_messages: List[Message] = []
                user_prompt_str = construct_user_prompt(workflow_state) # Uses current state including retry info
                current_turn_messages.append(Message(role="system", content=SYSTEM_PROMPT))
                current_turn_messages.append(Message(role="user", content=user_prompt_str))

                # LLM Call
                llm_response = await self.server.chat_completion(
                    messages=current_turn_messages, tools=self.tools, tool_choice="auto", n=1,
                    max_tokens=self.config.max_token_length, temperature=0.5
                )
                assistant_message_obj = llm_response.choices[0].message
                assistant_content = assistant_message_obj.content or ""
                assistant_tool_calls = []
                if hasattr(assistant_message_obj, 'tool_calls') and assistant_message_obj.tool_calls:
                    assistant_tool_calls = [
                        {"id": tc.id, "type": tc.type, "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                        for tc in assistant_message_obj.tool_calls
                    ]
                current_turn_messages.append(Message(role="assistant", content=assistant_content, tool_calls=assistant_tool_calls if assistant_tool_calls else None))

                # Tool Execution
                tool_error_for_retry_prompt = None
                if assistant_tool_calls:
                    tool_call_request = assistant_tool_calls[0]
                    tool_name = tool_call_request["function"]["name"]
                    try:
                        tool_args = json.loads(tool_call_request["function"]["arguments"])
                        tool_result = await self._execute_tool(tool_name, tool_args, workflow_state)
                        current_turn_messages.append(Message(role="tool", tool_call_id=tool_call_request["id"] , name=tool_name, content=json.dumps(tool_result)))
                        workflow_state["last_tool_success"] = tool_result.get("success", False)
                        if not workflow_state["last_tool_success"]:
                            tool_error_for_retry_prompt = tool_result.get("error", "Tool execution failed.")
                    except Exception as e:
                        error_msg = f"Error processing tool {tool_name}: {str(e)}"
                        current_turn_messages.append(Message(role="tool", tool_call_id=tool_call_request["id"], name=tool_name, content=error_msg))
                        workflow_state["last_tool_success"] = False
                        tool_error_for_retry_prompt = error_msg
                else: # No tool called
                    workflow_state["last_tool_success"] = False
                    expected_tool_name = {0:"AF2",1:"RFD",2:"PMPNN",3:"AF2M"}.get(workflow_state["current_internal_step"], "a tool")
                    tool_error_for_retry_prompt = f"No tool was called, but {expected_tool_name} was expected."

                workflow_state["previous_tool_error_message"] = tool_error_for_retry_prompt

                # Scoring and Accumulation for JSONL
                turn_score_details = self._score_trajectory(current_turn_messages, workflow_state)
                current_turn_reward = turn_score_details.get("overall_reward", 0.0)
                workflow_state["cumulative_reward"] += current_turn_reward

                tokenization_result = tokenize_for_trainer(self.tokenizer, current_turn_messages, include_messages=False)
                all_turns_data_for_jsonl.append({
                    "tokens_this_turn": tokenization_result["tokens"],
                    "masks_this_turn": tokenization_result["masks"],
                    "score_this_turn": current_turn_reward,
                    "messages_this_turn": current_turn_messages.copy(),
                    "overrides_this_turn": turn_score_details.copy()
                })

                # Workflow Progression / Retry Logic for process_mode
                if workflow_state["last_tool_success"]:
                    workflow_state["current_internal_step"] += 1
                    workflow_state["retry_count_this_internal_step"] = 0 # Reset for new step
                    workflow_state["previous_tool_error_message"] = None
                else: # Tool call failed or was incorrect
                    if workflow_state["current_internal_step"] <= 3: # Retry for steps 0, 1, 2, AND 3
                        workflow_state["retry_count_this_internal_step"] += 1
                        if workflow_state["retry_count_this_internal_step"] > self.config.max_retries_per_internal_step:
                            logger.warning(f"Workflow {item_id}, Step {workflow_state['current_internal_step']}: Max retries ({self.config.max_retries_per_internal_step}) reached. Terminating workflow for this item.")
                            workflow_state["workflow_complete_flag"] = True # Failed to progress
                            break # Exit the internal while loop
                        else:
                            logger.info(f"Workflow {item_id}, Step {workflow_state['current_internal_step']}: Failed, attempt {workflow_state['retry_count_this_internal_step']}. Retrying same step.")
                            # Loop continues, construct_user_prompt will use retry info
                    else: # Should never reach here with MAX_INTERNAL_STEPS = 4, but keeping for safety
                        logger.warning(f"Workflow {item_id}, Step {workflow_state['current_internal_step']}: Failure at non-retryable step. Terminating workflow.")
                        workflow_state["workflow_complete_flag"] = True
                        break # Exit the internal while loop

                if workflow_state["current_internal_step"] >= MAX_INTERNAL_STEPS:
                    workflow_state["workflow_complete_flag"] = True
                    logger.info(f"Workflow {item_id}: All internal steps completed successfully.")
                    # No break here, loop condition will handle it

            # After the internal while loop (for process mode)
            if not all_turns_data_for_jsonl:
                logger.warning(f"Workflow {item_id} in process mode: No turn data collected.")
                return None, []

            # --- Start of Fix for jsonl2html ---
            html_compatible_messages: List[str] = []
            html_compatible_scores: List[float] = []
            # `overrides_for_jsonl` will store the detailed scoring dict for each turn,
            # matching the structure of `html_compatible_messages` and `html_compatible_scores`.
            overrides_for_jsonl: List[Dict[str, Any]] = []


            for turn_idx, turn_data in enumerate(all_turns_data_for_jsonl):
                # Format messages for this turn into a single readable string
                turn_str_parts = [f"--- Workflow {item_id} - Turn {turn_idx + 1} ---"]
                if turn_data.get("messages_this_turn"):
                    for msg_obj in turn_data["messages_this_turn"]:
                        content_str = str(msg_obj.get("content", "[No Content]"))
                        if msg_obj.get("tool_calls"):
                            try:
                                tool_calls_str = json.dumps(msg_obj.get("tool_calls"), indent=2)
                                content_str += f"\nTool Calls:\n{tool_calls_str}"
                            except TypeError: # Handle non-serializable content if any
                                content_str += f"\nTool Calls: [Error serializing tool_calls]"
                        turn_str_parts.append(f"**{msg_obj.get('role', 'unknown').upper()}**: {content_str}")
                else:
                    turn_str_parts.append("No messages recorded for this turn.")

                html_compatible_messages.append("\n\n".join(turn_str_parts))

                # Get the score for this specific turn
                turn_score = turn_data.get("overrides_this_turn", {}).get("overall_reward", 0.0)
                html_compatible_scores.append(turn_score)

                # Add the detailed scoring dictionary for this turn
                overrides_for_jsonl.append(turn_data.get("overrides_this_turn", {}))


            final_workflow_reward = workflow_state.get("cumulative_reward", 0.0)
            # If the complex was evaluated successfully, the last turn's reward is the final one.
            if workflow_state.get("complex_evaluated") and workflow_state.get("last_tool_success"):
                 final_workflow_reward = all_turns_data_for_jsonl[-1].get("overrides_this_turn", {}).get("overall_reward", 0.0)

            # For the ScoredDataGroup that will be handled by BaseEnv
            # We need tokens and masks for each "message" (turn) if we want BaseEnv to consider it valid
            # For simplicity, we can just repeat the last turn's tokens/masks, or use placeholders
            # if actual per-turn tokens aren't critical for the JSONL's main purpose (which is visualization via messages/scores).
            # Let's create placeholder tokens/masks if full history isn't needed by the trainer for process_mode.
            # Or, better, store actual tokens for each turn if available.

            all_tokens_per_turn = [turn_data["tokens_this_turn"] for turn_data in all_turns_data_for_jsonl if turn_data.get("tokens_this_turn")]
            all_masks_per_turn = [turn_data["masks_this_turn"] for turn_data in all_turns_data_for_jsonl if turn_data.get("masks_this_turn")]

            # Ensure all_tokens_per_turn and all_masks_per_turn have same length as html_compatible_messages
            # If some turns didn't produce tokens (e.g. error), we might need to pad or handle.
            # For now, assuming all_turns_data_for_jsonl consistently has tokens/masks for each entry that contributes to html_compatible_messages.
            if len(all_tokens_per_turn) != len(html_compatible_messages):
                logger.error(f"CRITICAL: Mismatch between tokenized turns ({len(all_tokens_per_turn)}) and HTML messages ({len(html_compatible_messages)}). JSONL will be problematic.")
                # Fallback: repeat last turn's tokens if necessary, though this isn't ideal.
                if all_turns_data_for_jsonl and all_tokens_per_turn:
                    last_tokens = all_tokens_per_turn[-1]
                    last_masks = all_masks_per_turn[-1]
                    all_tokens_per_turn = [last_tokens] * len(html_compatible_messages)
                    all_masks_per_turn = [last_masks] * len(html_compatible_messages)
                else: # No token data at all
                    all_tokens_per_turn = [[] for _ in html_compatible_messages]
                    all_masks_per_turn = [[] for _ in html_compatible_messages]

            # This is the ScoredDataGroup that will be written to JSONL by BaseEnv
            process_mode_scored_data = ScoredDataGroup(
                tokens=all_tokens_per_turn,  # List of token lists, one for each turn
                masks=all_masks_per_turn,    # List of mask lists, one for each turn

                # These are critical for jsonl2html
                messages=html_compatible_messages, # List of strings, one per turn
                scores=html_compatible_scores,     # List of floats, one per turn

                # Store detailed overrides per turn, matching the length of messages/scores
                overrides=overrides_for_jsonl,

                group_overrides={
                    "group_size": len(html_compatible_messages), # Effective group size is number of turns
                    "item_id": item_id,
                    "is_process_mode_full_workflow": True,
                    "final_score_for_workflow": final_workflow_reward, # Store the overall workflow score here
                    "target_sequence": workflow_state.get("target_sequence", "N/A"),
                    "designed_binder_sequence": workflow_state.get("designed_binder_sequence", "N/A"),
                    "final_plddt": workflow_state.get("af2_multimer_plddt", 0.0)
                }
            )
            # --- End of Fix for jsonl2html ---

            # Log detailed workflow state to WandB (this call should use workflow_state directly)
            await self.add_rollouts_for_wandb(data_for_log=workflow_state.copy()) # Keep passing workflow_state for detailed wandb logging

            self.completed_episode_metrics.append(workflow_state.copy())
            if item_id in self.episodes_state: del self.episodes_state[item_id]
            return process_mode_scored_data, []

        else:
            # --- SERVE MODE: Process one turn, use backlog for continuation ---
            current_turn_messages_serve: List[Message] = []
            user_prompt_str_serve = construct_user_prompt(workflow_state) # Will include retry info if state reflects it
            current_turn_messages_serve.append(Message(role="system", content=SYSTEM_PROMPT))
            current_turn_messages_serve.append(Message(role="user", content=user_prompt_str_serve))

            llm_response_serve = await self.server.chat_completion(
                messages=current_turn_messages_serve, tools=self.tools, tool_choice="auto", n=1,
                max_tokens=self.config.max_token_length, temperature=0.5
            )
            assistant_message_obj_serve = llm_response_serve.choices[0].message
            assistant_content_serve = assistant_message_obj_serve.content or ""
            assistant_tool_calls_serve = []
            if hasattr(assistant_message_obj_serve, 'tool_calls') and assistant_message_obj_serve.tool_calls:
                assistant_tool_calls_serve = [
                    {"id": tc.id, "type": tc.type, "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in assistant_message_obj_serve.tool_calls
                ]
            current_turn_messages_serve.append(Message(role="assistant", content=assistant_content_serve, tool_calls=assistant_tool_calls_serve if assistant_tool_calls_serve else None))

            tool_error_for_retry_prompt_serve = None
            if assistant_tool_calls_serve:
                tool_call_request_serve = assistant_tool_calls_serve[0]
                tool_name_serve = tool_call_request_serve["function"]["name"]
                try:
                    tool_args_json_str = tool_call_request_serve["function"]["arguments"]
                    tool_args_serve = json.loads(tool_args_json_str)
                    tool_result_serve = await self._execute_tool(tool_name_serve, tool_args_serve, workflow_state)
                    current_turn_messages_serve.append(Message(role="tool", tool_call_id=tool_call_request_serve["id"] , name=tool_name_serve, content=json.dumps(tool_result_serve)))
                    workflow_state["last_tool_success"] = tool_result_serve.get("success", False)
                    if not workflow_state["last_tool_success"]:
                        tool_error_for_retry_prompt_serve = tool_result_serve.get("error", "Tool execution failed.")
                except Exception as e: # Catch JSONDecodeError and others
                    error_msg_serve = f"Error processing tool {tool_name_serve}: {str(e)}"
                    current_turn_messages_serve.append(Message(role="tool", tool_call_id=tool_call_request_serve["id"], name=tool_name_serve, content=error_msg_serve))
                    workflow_state["last_tool_success"] = False
                    tool_error_for_retry_prompt_serve = error_msg_serve
            else:
                workflow_state["last_tool_success"] = False
                expected_tool_name_serve = {0:"AF2",1:"RFD",2:"PMPNN",3:"AF2M"}.get(workflow_state["current_internal_step"], "a tool")
                tool_error_for_retry_prompt_serve = f"No tool was called, but {expected_tool_name_serve} was expected."

            workflow_state["previous_tool_error_message"] = tool_error_for_retry_prompt_serve

            turn_score_details_serve = self._score_trajectory(current_turn_messages_serve, workflow_state)
            current_turn_reward_serve = turn_score_details_serve.get("overall_reward", 0.0)
            workflow_state["cumulative_reward"] += current_turn_reward_serve
            workflow_state["turn_messages_history"].append(current_turn_messages_serve.copy())

            tokenization_result_serve = tokenize_for_trainer(
                self.tokenizer, current_turn_messages_serve, include_messages=self.config.include_messages
            )
            scored_data_serve = ScoredDataGroup(
                tokens=[tokenization_result_serve["tokens"]],
                masks=[tokenization_result_serve["masks"]],
                scores=[current_turn_reward_serve],
                messages=[current_turn_messages_serve] if self.config.include_messages else None,
                overrides=[turn_score_details_serve],
                group_overrides={"group_size": 1}  # Add group_overrides for serve mode too
            )

            backlog_items_serve = []
            if workflow_state["last_tool_success"]:
                workflow_state["current_internal_step"] += 1
                workflow_state["retry_count_this_internal_step"] = 0
                workflow_state["previous_tool_error_message"] = None
            else: # Tool failed or was incorrect
                if workflow_state["current_internal_step"] <= 3: # Retry for steps 0, 1, 2, AND 3
                    workflow_state["retry_count_this_internal_step"] += 1
                    if workflow_state["retry_count_this_internal_step"] > self.config.max_retries_per_internal_step:
                        logger.warning(f"Workflow {item_id}, Step {workflow_state['current_internal_step']} (Serve Mode): Max retries reached. Terminating.")
                        workflow_state["workflow_complete_flag"] = True
                    # else: it will be added to backlog below to retry
                else: # Failure at non-retryable step (should never reach here with MAX_INTERNAL_STEPS = 4)
                    logger.warning(f"Workflow {item_id}, Step {workflow_state['current_internal_step']} (Serve Mode): Failure at non-retryable step. Terminating.")
                    workflow_state["workflow_complete_flag"] = True

            if workflow_state["current_internal_step"] < 4 and not workflow_state.get("workflow_complete_flag"):
                # Add to backlog if:
                # 1. Last tool was successful (to move to next step)
                # OR
                # 2. Last tool failed, current step is <= 3, and we haven't hit max retries (to retry current step)
                should_add_to_backlog = False
                if workflow_state["last_tool_success"]:
                    should_add_to_backlog = True
                elif workflow_state["current_internal_step"] <= 3 and \
                     workflow_state["retry_count_this_internal_step"] <= self.config.max_retries_per_internal_step:
                    should_add_to_backlog = True

                if should_add_to_backlog:
                    backlog_items_serve.append(item_id)
                else: # Condition for adding to backlog not met
                    workflow_state["workflow_complete_flag"] = True # Mark as complete (due to failure beyond retries)
                    logger.info(f"Workflow for {item_id} (Serve Mode) not added to backlog and marked complete. Internal step: {workflow_state['current_internal_step']}")

            if workflow_state.get("workflow_complete_flag"): # If flag was set either by reaching step 4 or by retry logic
                # For completed workflows in serve mode, use direct logging with workflow_state
                # before it gets deleted
                if item_id in self.episodes_state:
                    # Use direct workflow_state logging for maximum detail
                    await self.add_rollouts_for_wandb(data_for_log=self.episodes_state[item_id].copy())
                    self.completed_episode_metrics.append(self.episodes_state[item_id].copy())
                    del self.episodes_state[item_id]
                # Note: We don't need to call add_rollouts_for_wandb with scored_data_serve here
                # BaseEnv.handle_send_to_api will call it automatically with the scored_data_serve
                # that we return

            return scored_data_serve, backlog_items_serve

    def _score_trajectory(self, turn_messages: List[Message], workflow_state: Dict) -> Dict[str, float]:
        """
        Scores a single turn's trajectory based on the specified reward logic.
        - Steps 0-2: Format reward (0.2 for correct & successful tool call, 0 otherwise).
        - Step 3 (AF2-Multimer): Reward based on pLDDT.
        """
        detailed_scores = {
            "overall_reward": 0.0,
            "raw_plddt": 0.0,
        }

        internal_step = workflow_state.get("current_internal_step")
        last_tool_success = workflow_state.get("last_tool_success", False)

        # ***** MODIFIED HERE *****
        assistant_msg_dict = next((m for m in reversed(turn_messages) if m.get("role") == "assistant"), None)

        expected_tool_for_step = {
            0: "predict_target_structure_alphafold2",
            1: "design_binder_backbone_rfdiffusion",
            2: "design_binder_sequence_proteinmpnn",
            3: "evaluate_binder_complex_alphafold2_multimer"
        }.get(internal_step)

        called_tool_name = None
        # ***** AND HERE *****
        if assistant_msg_dict and assistant_msg_dict.get("tool_calls"):
            tool_calls_list = assistant_msg_dict.get("tool_calls")
            if tool_calls_list and isinstance(tool_calls_list, list) and len(tool_calls_list) > 0:
                # Assuming tool_calls_list[0] is a dict as per your Message structure
                function_call_dict = tool_calls_list[0].get("function")
                if function_call_dict and isinstance(function_call_dict, dict):
                     called_tool_name = function_call_dict.get("name")

        # --- Scoring for Steps 0, 1, 2 (Internal Steps before AF2-Multimer) ---
        if internal_step < 3:
            if last_tool_success and called_tool_name == expected_tool_for_step:
                detailed_scores["overall_reward"] = 0.2
                logger.info(f"Workflow {workflow_state['item_id']}, Step {internal_step}: Correct tool '{called_tool_name}' used successfully. Reward: 0.2")
            else:
                detailed_scores["overall_reward"] = 0.0
                if not last_tool_success and called_tool_name:
                    logger.warning(f"Workflow {workflow_state['item_id']}, Step {internal_step}: Tool '{called_tool_name}' execution failed. Reward: 0.0")
                elif called_tool_name != expected_tool_for_step:
                    logger.warning(f"Workflow {workflow_state['item_id']}, Step {internal_step}: Incorrect tool '{called_tool_name}' used (expected '{expected_tool_for_step}'). Reward: 0.0")
                elif not called_tool_name and expected_tool_for_step:
                    logger.warning(f"Workflow {workflow_state['item_id']}, Step {internal_step}: No tool called, but expected '{expected_tool_for_step}'. Reward: 0.0")

        # --- Scoring for Step 3 (AF2-Multimer evaluation) ---
        elif internal_step == 3:
            if workflow_state.get("complex_evaluated") and last_tool_success and called_tool_name == expected_tool_for_step:
                plddt = workflow_state.get("af2_multimer_plddt", 0.0)
                detailed_scores["raw_plddt"] = plddt

                if plddt > 90.0:
                    detailed_scores["overall_reward"] = 1.0
                elif plddt > 50.0:
                    detailed_scores["overall_reward"] = 0.0 + (plddt - 50.0) * (1.0 - 0.0) / (90.0 - 50.0)
                    detailed_scores["overall_reward"] = max(0.0, min(detailed_scores["overall_reward"], 1.0))
                else:
                    detailed_scores["overall_reward"] = 0.0

                logger.info(f"Workflow {workflow_state['item_id']}, Step {internal_step} (AF2-Multimer): pLDDT={plddt:.2f}. Reward: {detailed_scores['overall_reward']:.2f}")
            else:
                detailed_scores["overall_reward"] = 0.0
                logger.warning(f"Workflow {workflow_state['item_id']}, Step {internal_step} (AF2-Multimer): Evaluation failed or wrong tool. Reward: -0.5. Last tool success: {last_tool_success}, Called: {called_tool_name}")

        else:
            logger.error(f"Workflow {workflow_state['item_id']}: Invalid internal_step {internal_step} in scoring.")
            detailed_scores["overall_reward"] = -1.0

        return detailed_scores

    async def postprocess_histories(
        self, trajectories: Optional[ScoredDataGroup]
    ) -> Optional[ScoredDataGroup]:
        """
        Post-processes a ScoredDataGroup for a single turn.
        Can be used for final adjustments or filtering if needed.
        """
        # Just pass through trajectories without modification
        return trajectories

    async def evaluate(self, *args, **kwargs):
        """
        Evaluate the environment's performance.
        This method is called periodically by the BaseEnv.env_manager.
        For BinderBenchEnv, it will aggregate metrics from completed workflows.
        """
        logger.info(f"Running evaluation for {self.name}...")
        if not self.completed_episode_metrics:
            logger.info("No completed episodes to evaluate since last evaluation.")
            self.eval_metrics = [] # Ensure eval_metrics is an empty list if no new data
            if self.config.use_wandb:
                await self.wandb_log({}) # Log that no eval data was present this cycle
            return

        # --- Metrics Calculation ---
        # These metrics are based on the episodes completed *since the last evaluation*
        # or since the start if this is the first evaluation.
        plddts, ptms, iptms, cumulative_rewards, workflow_successes = [], [], [], [], []

        # Use a copy of the buffer for this evaluation cycle
        current_eval_episodes = self.completed_episode_metrics.copy()
        # self.completed_episode_metrics.clear() # Clear the main buffer for the next cycle

        for ep_state in current_eval_episodes:
            if ep_state.get("complex_evaluated") and ep_state.get("last_tool_success"):
                plddts.append(ep_state.get("af2_multimer_plddt", 0.0))
                # ptms.append(ep_state.get("af2_multimer_ptm", 0.0)) # You set these to 0.0
                # iptms.append(ep_state.get("af2_multimer_iptm", 0.0))# You set these to 0.0
                workflow_successes.append(1.0)
            else:
                workflow_successes.append(0.0)
            cumulative_rewards.append(ep_state.get("cumulative_reward", 0.0))

        self.eval_metrics = [] # Reset class member for current evaluation results
        if plddts:
            self.eval_metrics.append(("eval/avg_plddt", sum(plddts) / len(plddts)))
        # if ptms: # Not currently being populated with real values
        #     self.eval_metrics.append(("eval/avg_ptm", sum(ptms) / len(ptms)))
        # if iptms: # Not currently being populated with real values
        #     self.eval_metrics.append(("eval/avg_iptm", sum(iptms) / len(iptms)))
        if cumulative_rewards:
            self.eval_metrics.append(("eval/avg_cumulative_reward", sum(cumulative_rewards) / len(cumulative_rewards)))
        if workflow_successes:
            self.eval_metrics.append(("eval/workflow_success_rate", sum(workflow_successes) / len(workflow_successes)))

        logger.info(f"Evaluation complete. Calculated metrics: {self.eval_metrics}")

        # Log to WandB immediately after evaluation if enabled
        if self.config.use_wandb:
            # self.wandb_log will pick up self.eval_metrics
            await self.wandb_log({})

        # It's important to clear self.completed_episode_metrics *after* they've been processed
        # for this eval cycle to avoid re-evaluating old data.
        # If evaluation is meant to be on *all* completed episodes ever, don't clear.
        # Typically, eval is on data since last eval or a fixed test set.
        # Given it's populated by collect_trajectories, clearing seems appropriate for periodic eval.
        self.completed_episode_metrics.clear()

    async def add_rollouts_for_wandb(self,
                                 scored_data_group: ScoredDataGroup = None,  # From BaseEnv
                                 item_id: Item = None,  # From BaseEnv
                                 data_for_log: Dict = None):  # Our custom param for direct workflow_state logging
        """Adds a workflow summary to the wandb rollout buffer.

        This method has two modes of operation:
        1. Direct logging with workflow_state (preferred for detailed logging):
           - Called from within collect_trajectories with data_for_log=workflow_state.copy()
           - This provides maximum detail for logging

        2. BaseEnv compatibility mode:
           - Called from BaseEnv.handle_send_to_api with scored_data_group and item_id
           - Used automatically by the framework
           - May have less detail if workflow_state was already deleted

        Args:
            scored_data_group: The ScoredDataGroup containing token, mask, and score data (from BaseEnv)
            item_id: The item identifier, which is the key to our episodes_state (from BaseEnv)
            data_for_log: Direct workflow state to log (our custom parameter for direct logging)
        """
        if not self.config.use_wandb or not hasattr(self, "rollouts_for_wandb"):
            # Ensure rollouts_for_wandb exists
            if not hasattr(self, "rollouts_for_wandb"):
                self.rollouts_for_wandb = []

        # Determine the workflow state to use
        workflow_state = None

        # Case 1: Direct workflow state provided (most detailed)
        if data_for_log is not None and isinstance(data_for_log, dict):
            workflow_state = data_for_log
            # Extract item_id from data_for_log if needed
            if item_id is None and "item_id" in workflow_state:
                item_id = workflow_state["item_id"]

        # Case 2: Try to get workflow_state from episodes_state (if not already deleted)
        elif item_id is not None and item_id in self.episodes_state:
            workflow_state = self.episodes_state[item_id]

        # Case 3: No usable state - early return with a debug log (not warning)
        # This happens in BaseEnv.handle_send_to_api after workflow is already completed
        if workflow_state is None:
            # This is expected in BaseEnv's call after workflow_state is deleted, so use debug level
            logger.debug(f"No workflow_state available for WandB logging (item_id={item_id})")
            return

        # Customize what you want to see in the WandB table for a completed workflow
        # Handle cases where values might be None
        target_seq = workflow_state.get("target_sequence", "N/A")

        # Handle designed_binder which might be None
        designed_binder = workflow_state.get("designed_binder_sequence", "N/A")
        if designed_binder is None:
            designed_binder = "N/A"

        plddt = workflow_state.get("af2_multimer_plddt", 0.0)
        iptm = workflow_state.get("af2_multimer_iptm", 0.0) # Even if 0, log it
        cumulative_reward = workflow_state.get("cumulative_reward", 0.0)

        # For messages, maybe just the final assistant message that led to AF2M or a summary
        # Storing all turn_messages_history can make the table huge.
        # Let's take the last turn's messages for this example.
        last_turn_messages_str = "No messages."
        try:
            if workflow_state.get("turn_messages_history") and len(workflow_state["turn_messages_history"]) > 0:
                last_turn_convo = workflow_state["turn_messages_history"][-1]
                last_turn_messages_str = "\n---\n".join(
                    [f"{msg.get('role', 'unknown')}: {str(msg.get('content', ''))[:200]}..." for msg in last_turn_convo]
                )
        except Exception as e:
            logger.error(f"Error processing messages for WandB: {e}")
            last_turn_messages_str = "Error processing messages"

        # Safely truncate strings
        target_preview = target_seq[:30] + "..." if isinstance(target_seq, str) and len(target_seq) > 30 else target_seq

        if designed_binder == "N/A" or designed_binder is None:
            binder_preview = "N/A"
        else:
            binder_preview = designed_binder[:30] + "..." if len(str(designed_binder)) > 30 else designed_binder

        # Use item_id from workflow_state if still None
        if item_id is None:
            item_id = workflow_state.get("item_id", "unknown-id")

        # Add to rollouts buffer
        self.rollouts_for_wandb.append(
            ( # This tuple structure will be used by create_rollout_table
                str(item_id),  # Ensure item_id is a string
                target_preview,
                binder_preview,
                f"{plddt:.2f}",
                f"{iptm:.2f}",
                f"{cumulative_reward:.3f}",
                last_turn_messages_str # Or a link to the full JSONL entry, or more details
            )
        )
        if len(self.rollouts_for_wandb) > self.config.num_rollouts_to_keep:
            self.rollouts_for_wandb.pop(0)

    async def create_rollout_table(self, wandb_metrics: Dict) -> Dict:
        """Creates a wandb.Table from the buffered rollouts."""
        if hasattr(self, "rollouts_for_wandb") and self.rollouts_for_wandb:
            # Define columns based on what add_rollouts_for_wandb appends
            columns = ["Item ID", "Target (Preview)", "Designed Binder (Preview)",
                      "Final pLDDT", "Final ipTM", "Cumulative Reward", "Last Turn Messages"]
            table = wandb.Table(columns=columns)
            for rollout_tuple in self.rollouts_for_wandb:
                table.add_data(*rollout_tuple) # Unpack the tuple

            # Use a unique key for the table, prepended by wandb_prepend
            table_key = f"env_rollouts/{self.wandb_prepend}/completed_workflows"
            if self.wandb_prepend is None and hasattr(self, "name"): # Fallback if wandb_prepend not set yet
                 table_key = f"env_rollouts/{self.name}/completed_workflows"

            wandb_metrics[table_key] = table
            self.rollouts_for_wandb.clear()
        return wandb_metrics

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        if wandb_metrics is None:
            wandb_metrics = {}

        # Create and add the rollout table to wandb_metrics
        if hasattr(self, "rollouts_for_wandb") and self.rollouts_for_wandb:
            wandb_metrics = await self.create_rollout_table(wandb_metrics)

        # Add any training-time aggregated metrics (not from self.completed_episode_metrics,
        # as that's now handled by evaluate for eval-specific logging)
        # For example, if you had a buffer for per-turn scores during training rollouts:
        # if self.per_turn_score_buffer:
        #     wandb_metrics[f"train/{self.wandb_prepend}/avg_turn_reward"] = sum(self.per_turn_score_buffer) / len(self.per_turn_score_buffer)
        #     self.per_turn_score_buffer.clear()

        # The self.eval_metrics (populated by evaluate()) will be picked up by super().wandb_log()
        await super().wandb_log(wandb_metrics)

if __name__ == "__main__":
    BinderBenchEnv.cli()
