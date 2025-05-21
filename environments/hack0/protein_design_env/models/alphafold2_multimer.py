import os
import logging
import aiohttp
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_URL = "https://health.api.nvidia.com/v1/biology/deepmind/alphafold2-multimer"
DEFAULT_STATUS_URL = "https://health.api.nvidia.com/v1/status"

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

async def _process_pdb_and_scores_from_api(
    pdb_contents: List[str],
    job_id: str,
    api_response_json: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Path] = None
) -> Optional[Dict[str, Any]]:
    """
    Processes a list of PDB strings received from the API JSON response.
    - The API responds with a direct list of PDB strings, not a nested JSON structure
    - This function saves each PDB to disk and calculates pLDDT scores
    - The api_response_json parameter is for potential future use if the API adds metadata
    - The output_dir parameter allows for customizing where files are saved
    """
    if output_dir is None:
        # Default behavior if no output dir is provided
        output_dir_name = f"alphafold2_multimer_output_{job_id}_results"
        output_dir = Path(f"./{output_dir_name}")
    
    # Make sure the directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving AlphaFold2-Multimer results for job {job_id} to directory: {output_dir}")

    results: Dict[str, Any] = {
        "output_directory": str(output_dir),
        "structures": [],
        "ptm_score": None,
        "iptm_score": None,
    }

    if isinstance(api_response_json, dict):
        logger.info(f"Attempting to extract additional scores from API JSON response for job {job_id}.")
        results["ptm_score"] = api_response_json.get("ptm")
        results["iptm_score"] = api_response_json.get("iptm")
        if results["ptm_score"] is not None or results["iptm_score"] is not None:
            logger.info(f"Extracted from API JSON: pTM={results['ptm_score']}, ipTM={results['iptm_score']}")
    else:
        logger.info(f"No additional API JSON dictionary provided or it's not a dict for job {job_id}; ptm/iptm scores will be None unless found elsewhere.")

    if not pdb_contents or not isinstance(pdb_contents, list) or not all(isinstance(s, str) for s in pdb_contents):
        logger.warning(f"No valid PDB content strings provided for job {job_id}.")
        return results # Return with empty structures list

    logger.info(f"Processing {len(pdb_contents)} PDB structure(s) for job {job_id}.")

    for i, pdb_str in enumerate(pdb_contents):
        if not pdb_str.strip():
            logger.debug(f"Skipping empty PDB string at index {i} for job {job_id}.")
            continue

        structure_data: Dict[str, Any] = {
            "model_index": i,
            "pdb_content": pdb_str
        }

        avg_plddt, plddts_per_ca_residue, plddts_by_chain = calculate_plddt_from_pdb_string(pdb_str)

        structure_data["average_plddt"] = avg_plddt
        structure_data["plddt_scores_per_ca_residue"] = plddts_per_ca_residue
        structure_data["plddt_scores_per_chain"] = plddts_by_chain

        avg_plddt_per_chain = {}
        for chain_id, chain_plddts in plddts_by_chain.items():
            if chain_plddts:
                avg_plddt_per_chain[chain_id] = sum(chain_plddts) / len(chain_plddts)
            else:
                avg_plddt_per_chain[chain_id] = 0.0
        structure_data["average_plddt_per_chain"] = avg_plddt_per_chain

        pdb_file_name_stem = f"alphafold2_multimer_output_{job_id}"
        rank_suffix = f"_model_{i+1}"
        pdb_file_path = output_dir / f"{pdb_file_name_stem}{rank_suffix}.pdb"

        try:
            with open(pdb_file_path, "w", encoding='utf-8') as f_pdb:
                f_pdb.write(pdb_str)
            structure_data["saved_pdb_path"] = str(pdb_file_path)
            logger.info(f"Saved PDB model {i+1} for job {job_id} to {pdb_file_path} with overall avg_pLDDT: {avg_plddt:.2f}")
        except Exception as e_write:
            logger.error(f"Failed to write PDB file {pdb_file_path} for job {job_id}: {e_write}")
            structure_data["saved_pdb_path"] = None

        results["structures"].append(structure_data)

    if results["structures"]:
         logger.info(f"Successfully processed and calculated pLDDTs for {len(results['structures'])} structures for job {job_id}.")

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
    timeout: int = 3600,
    output_dir: Optional[Path] = None
) -> Optional[Dict[str, Any]]:
    """
    Call the NVIDIA NIM AlphaFold2-Multimer API.
    The API now returns JSON with a list of PDB strings, which we process to calculate pLDDT scores.
    Returns a dictionary with processed PDB strings and computed scores.
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
                    logger.info(f"SYNC Final response headers: {response.headers}")
                    content_type = response.headers.get("Content-Type", "").lower()
                    logger.info(f"SYNC Final response content-type: {content_type}")

                    if "application/json" in content_type:
                        api_response_json_payload = await response.json()
                        if not isinstance(api_response_json_payload, list):
                            return {"success": False, "error": "Sync JSON response not a list of PDBs as expected."}
                        req_id_sync = response.headers.get("nvcf-reqid", "sync_job") # Get req_id or make one up
                        return await _process_pdb_and_scores_from_api(
                            pdb_contents=api_response_json_payload,
                            job_id=req_id_sync,
                            api_response_json=None,
                            output_dir=output_dir
                        )
                    else:
                        return {"success": False, "error": f"Sync response unexpected content type: {content_type}"}
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
                        return {"success": False, "error": "No request ID in 202 response headers"}
                else:
                    logger.error(f"Error calling AlphaFold2-Multimer API (POST): {response.status}")
                    text = await response.text()
                    logger.error(f"Response: {text}")
                    return {"success": False, "error": f"Error calling API: {response.status}", "detail": text}
    except asyncio.TimeoutError:
        logger.error(f"Timeout during AlphaFold2-Multimer API (initial POST).")
        return {"success": False, "error": "Timeout during initial API request"}
    except Exception as e:
        logger.error(f"Exception during AlphaFold2-Multimer API call (initial POST): {e}", exc_info=True)
        return {"success": False, "error": f"Exception during API call: {str(e)}"}

async def _poll_job_status(
    req_id: str,
    headers: Dict[str, str],
    status_url: str,
    polling_interval: int = 30,
    overall_timeout: int = 3600
) -> Optional[Dict[str, Any]]:
    start_time = asyncio.get_event_loop().time()
    per_status_request_timeout = 600
    logger.info(f"Polling job {req_id}. Individual status check timeout: {per_status_request_timeout}s, Polling interval: {polling_interval}s, Overall timeout: {overall_timeout}s")

    while True:
        current_loop_time = asyncio.get_event_loop().time()
        elapsed_time = current_loop_time - start_time

        if elapsed_time >= overall_timeout:
            logger.error(f"Overall polling timeout of {overall_timeout}s exceeded for job {req_id}.")
            return {"success": False, "error": "Overall polling timeout exceeded."}

        remaining_time_for_overall_timeout = overall_timeout - elapsed_time
        current_status_check_timeout = min(per_status_request_timeout, remaining_time_for_overall_timeout)

        if current_status_check_timeout <= 0:
             logger.error(f"Not enough time left for another status check for job {req_id} within overall_timeout.")
             return {"success": False, "error": "Not enough time for status check within overall timeout."}

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

                        if response.content_type == 'application/json':
                            try:
                                api_response_json_payload = await response.json()
                                logger.debug(f"API JSON Response for job {req_id}: {str(api_response_json_payload)[:500]}...")

                                if not isinstance(api_response_json_payload, list):
                                    logger.error(f"Job {req_id}: Expected API response to be a list of PDB strings, got {type(api_response_json_payload)}.")
                                    return {"success": False, "error": "API response was not a list of PDB strings."}

                                return await _process_pdb_and_scores_from_api(
                                    pdb_contents=api_response_json_payload,
                                    job_id=req_id,
                                    api_response_json=None,
                                    output_dir=output_dir
                                )
                            except json.JSONDecodeError:
                                logger.error(f"Job {req_id}: Failed to decode JSON response from API.", exc_info=True)
                                raw_text = await response.text()
                                logger.debug(f"Raw text response: {raw_text[:500]}")
                                return {"success": False, "error": "Failed to decode JSON response."}
                        else:
                            logger.error(f"Job {req_id}: Unexpected content type {response.content_type}. Expected application/json.")
                            raw_text = await response.text()
                            logger.debug(f"Raw text response: {raw_text[:500]}")
                            return {"success": False, "error": f"Unexpected content type: {response.content_type}"}

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
                    else:
                        text = await response.text()
                        logger.error(f"Error checking AlphaFold2-Multimer job status {req_id}: HTTP {response.status} - {text}")
                        return {"success": False, "error": f"Status check failed with HTTP {response.status}", "detail": text}
        except asyncio.TimeoutError:
            logger.warning(f"Client-side timeout ({current_status_check_timeout}s) during status check for job {req_id}. Retrying poll after {polling_interval}s sleep.")
            await asyncio.sleep(polling_interval)
        except aiohttp.ClientError as e:
            logger.error(f"Client error polling job status for {req_id}: {e}. Retrying poll after {polling_interval}s.", exc_info=True)
            await asyncio.sleep(polling_interval)
        except Exception as e:
            logger.error(f"Unexpected error polling job status {req_id}: {e}", exc_info=True)
            return {"success": False, "error": f"Unexpected polling error: {str(e)}"}
