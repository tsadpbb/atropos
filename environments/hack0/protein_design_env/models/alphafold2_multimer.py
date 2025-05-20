"""AlphaFold2-Multimer API integration for NVIDIA NIM."""

import os
import logging
import aiohttp
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import zipfile
import io

logger = logging.getLogger(__name__)

# Default URL
DEFAULT_URL = "https://health.api.nvidia.com/v1/biology/deepmind/alphafold2-multimer"
DEFAULT_STATUS_URL = "https://health.api.nvidia.com/v1/status"

# Helper functions
def _split_pdb_content(concatenated_pdb_str: str) -> List[str]:
    """
    Splits a string containing concatenated PDB file contents.
    Assumes models are separated by "ENDMDL" or just "END" for the last/single model.
    """
    pdbs = []
    current_pdb_lines = []
    if not concatenated_pdb_str:
        return []

    for line in concatenated_pdb_str.splitlines(keepends=True):
        current_pdb_lines.append(line)
        if line.startswith("ENDMDL") or line.startswith("END "):
            pdbs.append("".join(current_pdb_lines).strip())
            current_pdb_lines = []
    
    if current_pdb_lines:
        remaining_pdb = "".join(current_pdb_lines).strip()
        if remaining_pdb:
            pdbs.append(remaining_pdb)
            
    return [pdb for pdb in pdbs if pdb]


def calculate_plddt_from_pdb_string(pdb_string: str) -> Tuple[float, List[float], Dict[str, List[float]]]:
    """
    Calculates the average pLDDT score from a PDB string for C-alpha atoms.
    Also returns a list of all C-alpha pLDDTs and a dictionary of pLDDTs per chain.

    Returns:
        A tuple containing:
        - average_plddt (float): Average pLDDT over all C-alpha atoms.
        - plddt_scores_per_ca (List[float]): List of pLDDTs for each C-alpha atom.
        - plddt_scores_per_chain (Dict[str, List[float]]): Dict mapping chain ID to its C-alpha pLDDTs.
    """
    total_plddt = 0.0
    ca_atom_count = 0
    plddt_scores_per_ca: List[float] = []
    plddt_scores_per_chain: Dict[str, List[float]] = {}

    for line in pdb_string.splitlines():
        if line.startswith("ATOM"):
            atom_name = line[12:16].strip()
            if atom_name == "CA":
                try:
                    plddt_value = float(line[60:66].strip())
                    total_plddt += plddt_value
                    plddt_scores_per_ca.append(plddt_value)
                    ca_atom_count += 1

                    chain_id = line[21:22].strip()
                    if chain_id not in plddt_scores_per_chain:
                        plddt_scores_per_chain[chain_id] = []
                    plddt_scores_per_chain[chain_id].append(plddt_value)

                except ValueError:
                    pass 
                except IndexError:
                    pass
    
    if ca_atom_count == 0:
        return 0.0, [], {}
    
    average_plddt = total_plddt / ca_atom_count
    return average_plddt, plddt_scores_per_ca, plddt_scores_per_chain

async def _process_nvidia_zip_output(
    zip_content: bytes,
    output_prefix: str,
    response_headers: Optional[Dict[str, str]] = None
) -> Optional[Dict[str, Any]]:
    """
    Processes the ZIP file content from NVIDIA NIM.
    - Expects a .response file with concatenated PDBs, or individual PDB files.
    - Extracts PDBs and calculates pLDDT scores for each structure.
    - Returns paths to saved files and calculated pLDDT scores.
    """
    output_dir = Path(f"./{output_prefix}_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the original ZIP file
    zip_file_path = output_dir / f"{output_prefix}.zip"
    with open(zip_file_path, 'wb') as f:
        f.write(zip_content)
    logger.info(f"Downloaded and saved original ZIP file to {zip_file_path}")

    # Initialize the results dictionary that call_alphafold2_multimer will return
    results: Dict[str, Any] = {
        "zip_file_path": str(zip_file_path), # Path to the saved original ZIP
        "structures": [], # List to hold info for each PDB structure found
        # We will NOT be trying to parse iptm/ptm from ranking_debug.json
        "iptm_score": None, # Explicitly None, or remove if not needed at all
        "ptm_score": None,  # Explicitly None, or remove
        # Optional: Store path to the .response file if it exists and is processed
        "extracted_response_file_path": None, 
    }

    pdb_strings_to_process = []

    try:
        with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
            # First, check for a ".response" file with concatenated PDBs
            response_file_name = None
            for member_name in zf.namelist():
                if member_name.lower().endswith(".response"):
                    response_file_name = member_name
                    break
            
            if response_file_name:
                logger.info(f"Found concatenated response file in ZIP: {response_file_name}")
                response_data = zf.read(response_file_name)
                response_content_str = response_data.decode('utf-8', errors='replace')
                
                # Save the raw .response file
                extracted_response_file_path = output_dir / Path(response_file_name).name
                with open(extracted_response_file_path, 'w', encoding='utf-8', errors='replace') as f_resp:
                    f_resp.write(response_content_str)
                results["extracted_response_file_path"] = str(extracted_response_file_path)
                logger.info(f"Saved raw content of '{response_file_name}' to {extracted_response_file_path}")
                
                pdb_strings_to_process.extend(_split_pdb_content(response_content_str))
            else:
                # If no ".response" file, look for individual PDB files
                logger.info("No .response file found. Looking for individual .pdb files in ZIP.")
                for member_name in zf.namelist():
                    if member_name.lower().endswith(".pdb"):
                        logger.info(f"Found individual PDB file in ZIP: {member_name}")
                        pdb_content_bytes = zf.read(member_name)
                        pdb_strings_to_process.append(pdb_content_bytes.decode('utf-8', errors='replace'))
            
            if not pdb_strings_to_process:
                logger.warning(f"No PDB content found in ZIP archive {zip_file_path} (either as .response or individual .pdb files).")
                return results # Return with empty structures list

            logger.info(f"Found {len(pdb_strings_to_process)} PDB structure(s) to process.")
            
            for i, pdb_str in enumerate(pdb_strings_to_process):
                if not pdb_str.strip(): # Skip empty PDB strings
                    logger.debug(f"Skipping empty PDB string at index {i}.")
                    continue

                structure_data: Dict[str, Any] = {
                    "model_index": i, # 0-indexed based on order found
                    "pdb_content": pdb_str # Store the raw PDB string
                }
                
                # Calculate pLDDT scores using your existing function
                avg_plddt, plddts_per_ca_residue, plddts_by_chain = calculate_plddt_from_pdb_string(pdb_str)
                
                structure_data["average_plddt"] = avg_plddt
                structure_data["plddt_scores_per_ca_residue"] = plddts_per_ca_residue # List of pLDDTs for CAs
                structure_data["plddt_scores_per_chain"] = plddts_by_chain # Dict: chain_id -> List[pLDDT]
                
                # Calculate average pLDDT for each chain (already in your previous code, good to keep)
                avg_plddt_per_chain = {}
                for chain_id, chain_plddts in plddts_by_chain.items():
                    if chain_plddts: # Avoid division by zero
                        avg_plddt_per_chain[chain_id] = sum(chain_plddts) / len(chain_plddts)
                    else:
                        avg_plddt_per_chain[chain_id] = 0.0
                structure_data["average_plddt_per_chain"] = avg_plddt_per_chain

                # Save the individual PDB string to a file
                pdb_file_name_stem = Path(output_prefix).stem
                # Suffix for rank if multiple models found, otherwise simpler name
                rank_suffix = f"_model_{i}" # Consistent naming for multiple models
                pdb_file_path = output_dir / f"{pdb_file_name_stem}{rank_suffix}.pdb"
                
                try:
                    with open(pdb_file_path, "w", encoding='utf-8') as f_pdb:
                        f_pdb.write(pdb_str)
                    structure_data["saved_pdb_path"] = str(pdb_file_path)
                    logger.info(f"Saved PDB model {i} to {pdb_file_path} with overall avg_pLDDT: {avg_plddt:.2f}")
                except Exception as e_write:
                    logger.error(f"Failed to write PDB file {pdb_file_path}: {e_write}")
                    structure_data["saved_pdb_path"] = None
                
                results["structures"].append(structure_data)

            if results["structures"]:
                 logger.info(f"Successfully processed and calculated pLDDTs for {len(results['structures'])} structures.")
        
    except zipfile.BadZipFile:
        logger.error(f"Failed to process ZIP: {zip_file_path} is not a valid ZIP file.")
        # results dictionary is already initialized, will be returned as is, potentially empty.
    except Exception as e:
        logger.error(f"An error occurred during ZIP processing of {zip_file_path}: {e}", exc_info=True)
        # As above, return results which might be partially filled or empty.
        
    return results

async def call_alphafold2_multimer(
    sequences: List[str],
    api_key: str,
    algorithm: str = "jackhmmer",
    e_value: float = 0.0001,
    iterations: int = 1,
    databases: List[str] = ["uniref90", "small_bfd", "mgnify"],
    relax_prediction: bool = True,
    selected_models: Optional[List[int]] = None,
    url: str = DEFAULT_URL,
    status_url: str = DEFAULT_STATUS_URL,
    polling_interval: int = 30,
    timeout: int = 3600
) -> Optional[Dict[str, Any]]:
    """
    Call the NVIDIA NIM AlphaFold2-Multimer API.
    Returns a dictionary structured by _process_nvidia_zip_output.
    """
    headers = {
        "content-type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "NVCF-POLL-SECONDS": "300", 
    }
    data: Dict[str, Any] = {
        "sequences": sequences,
        "algorithm": algorithm,
        "e_value": e_value,
        "iterations": iterations,
        "databases": databases,
        "relax_prediction": relax_prediction
    }
    if selected_models is not None:
        data["selected_models"] = selected_models
        logger.info(f"Using selected_models: {selected_models}")
    
    try:
        initial_post_timeout = min(timeout, 600) 
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=data,
                headers=headers,
                timeout=initial_post_timeout
            ) as response:
                if response.status == 200: 
                    logger.info("AlphaFold2-Multimer job completed synchronously.")
                    content = await response.read()
                    return await _process_nvidia_zip_output(
                        zip_content=content,
                        output_prefix="alphafold2_multimer_sync_output",
                        response_headers=response.headers 
                    )
                elif response.status == 202: 
                    req_id = response.headers.get("nvcf-reqid")
                    if req_id:
                        logger.info(f"AlphaFold2-Multimer job submitted, request ID: {req_id}")
                        return await _poll_job_status(
                            req_id=req_id,
                            headers=headers,
                            status_url=status_url,
                            polling_interval=polling_interval,
                            overall_timeout=timeout
                        )
                    else:
                        logger.error("No request ID in 202 response headers")
                        return None
                else: # Handle other error statuses from POST
                    logger.error(f"Error calling AlphaFold2-Multimer API (POST): {response.status}")
                    text = await response.text()
                    logger.error(f"Response: {text}")
                    return None
    except asyncio.TimeoutError:
        logger.error(f"Timeout during AlphaFold2-Multimer API (initial POST).")
        return None
    except Exception as e:
        logger.error(f"Exception during AlphaFold2-Multimer API call (initial POST): {e}", exc_info=True)
        return None
    
async def _poll_job_status(
    req_id: str,
    headers: Dict[str, str],
    status_url: str,
    polling_interval: int = 30, 
    overall_timeout: int = 3600 
) -> Optional[Dict[str, Any]]:
    start_time = asyncio.get_event_loop().time()
    # Allow status checks to wait longer, e.g., slightly more than NVCF-POLL-SECONDS if you use it,
    # or a fixed reasonably long duration.
    # The NVCF-POLL-SECONDS in the POST header is 300s.
    # The GET request to /status should also ideally respect a similar long-poll duration from the server.
    status_check_timeout = 330 # seconds (e.g., 5.5 minutes)
    logger.info(f"Polling job {req_id}. Status check timeout: {status_check_timeout}s, Polling interval: {polling_interval}s, Overall timeout: {overall_timeout}s")

    while True:
        current_loop_time = asyncio.get_event_loop().time()
        elapsed_time = current_loop_time - start_time
        
        if elapsed_time > overall_timeout:
            logger.error(f"Overall polling timeout of {overall_timeout}s exceeded for job {req_id}.")
            return None
        
        remaining_time_for_overall_timeout = overall_timeout - elapsed_time
        current_status_check_timeout = min(status_check_timeout, remaining_time_for_overall_timeout)
        if current_status_check_timeout <= 0:
             logger.error(f"Not enough time left for another status check for job {req_id} within overall_timeout.")
             return None

        try:
            async with aiohttp.ClientSession() as session:
                logger.debug(f"Checking status for {req_id} with timeout {current_status_check_timeout}s.")
                async with session.get(
                    f"{status_url}/{req_id}",
                    headers=headers,
                    timeout=current_status_check_timeout 
                ) as response:
                    if response.status == 200:
                        logger.info(f"AlphaFold2-Multimer job {req_id} completed (status 200).")
                        logger.info(f"FINAL 200 OK Response Headers for job {req_id}: {response.headers}")
                        logger.info(f"FINAL 200 OK Content-Type for job {req_id}: {response.content_type}")
                        zip_content_bytes = await response.read()
                        return await _process_nvidia_zip_output(
                            zip_content=zip_content_bytes,
                            output_prefix=f"alphafold2_multimer_output_{req_id}",
                            response_headers=response.headers
                        )
                    elif response.status == 202:
                        try:
                            job_status_json = await response.json() 
                            percent_complete = job_status_json.get('percentComplete', 'N/A')
                            status_message = job_status_json.get('status', 'running')
                            logger.debug(
                                f"Job {req_id} status: {status_message} ({percent_complete}% complete). Polling again in {polling_interval}s."
                            )
                        except (aiohttp.ContentTypeError, json.JSONDecodeError): 
                            logger.debug(
                                f"Job {req_id} still running (202 status, non-JSON/malformed JSON body). Polling again in {polling_interval}s."
                            )
                        await asyncio.sleep(polling_interval)
                    else: # Handle other error statuses from status GET
                        logger.error(f"Error checking AlphaFold2-Multimer job status {req_id}: {response.status}")
                        text = await response.text()
                        logger.error(f"Response: {text}")
                        # Log the error, but continue polling unless it's a fatal client error (4xx other than 429)
                        # or if the server explicitly indicates failure (e.g. 500, or a 200 with error status in body)
                        # For a 504 like you saw, we might want to retry a few times then give up.
                        # For now, this will return None on non-200/202, which your test script will catch.
                        return None 
        except asyncio.TimeoutError: 
            logger.warning(f"Client-side timeout ({current_status_check_timeout}s) during status check for job {req_id}. Retrying poll after {polling_interval}s sleep.")
            await asyncio.sleep(polling_interval) 
        except aiohttp.ClientError as e: 
            logger.error(f"Client error polling job status for {req_id}: {e}. Retrying poll after {polling_interval}s.", exc_info=True)
            await asyncio.sleep(polling_interval)
        except Exception as e: 
            logger.error(f"Unexpected error polling job status {req_id}: {e}", exc_info=True)
            return None