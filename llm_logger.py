"""
LLM Logger - Observability for LLM Requests

Logs all LLM requests and responses to files for debugging.
You cannot debug LLM interactions in the dark.

Usage:
    from djinnite.llm_logger import LLMLogger
    
    logger = LLMLogger("cost_estimation")
    request_id = logger.log_request(prompt, system_prompt)
    # ... call LLM ...
    logger.log_response(request_id, response_content, success=True)
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import uuid


class LLMLogger:
    """
    Logs LLM requests and responses to files for debugging.
    
    Creates two files per request:
    - request_{id}.txt - The rendered prompt sent to the LLM
    - response_{id}.txt - The raw response from the LLM
    """
    
    def __init__(self, task_name: str, log_dir: Optional[Path] = None):
        """
        Initialize the logger.
        
        Args:
            task_name: Name of the task (used in log directory)
            log_dir: Custom log directory (default: logs/llm/{task_name})
        """
        self.task_name = task_name
        
        if log_dir is None:
            # Default to <project_root>/logs/llm/{task_name}
            # project_root is the parent of the djinnite package
            project_root = Path(__file__).parent.parent
            log_dir = project_root / "logs" / "llm" / task_name
        
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Session tracking
        self.session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.request_count = 0
    
    def log_request(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> str:
        """
        Log an LLM request before sending.
        
        Args:
            prompt: The rendered prompt being sent
            system_prompt: Optional system prompt
            model: Model being used
            provider: Provider being used
            metadata: Additional metadata to log
            
        Returns:
            Request ID for correlating with response
        """
        self.request_count += 1
        request_id = f"{self.session_id}_{self.request_count:03d}"
        
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Build request log content
        content = f"""=== LLM REQUEST ===
Request ID: {request_id}
Timestamp: {timestamp}
Task: {self.task_name}
Provider: {provider or 'unknown'}
Model: {model or 'unknown'}

"""
        
        if system_prompt:
            content += f"--- SYSTEM PROMPT ---\n{system_prompt}\n\n"
        
        content += f"--- USER PROMPT ---\n{prompt}\n"
        
        if metadata:
            content += f"\n--- METADATA ---\n{json.dumps(metadata, indent=2)}\n"
        
        # Write to file
        request_file = self.log_dir / f"request_{request_id}.txt"
        request_file.write_text(content, encoding="utf-8")
        
        return request_id
    
    def log_response(
        self,
        request_id: str,
        response_content: str,
        success: bool = True,
        error: Optional[str] = None,
        usage: Optional[dict] = None,
        parsed_result: Optional[any] = None
    ) -> None:
        """
        Log an LLM response after receiving.
        
        Args:
            request_id: The request ID from log_request
            response_content: Raw response content from LLM
            success: Whether the response was successfully parsed
            error: Error message if parsing failed
            usage: Token usage statistics
            parsed_result: The parsed result (if successful)
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        
        status = "SUCCESS" if success else "FAILED"
        
        content = f"""=== LLM RESPONSE ===
Request ID: {request_id}
Timestamp: {timestamp}
Status: {status}

"""
        
        if error:
            content += f"--- ERROR ---\n{error}\n\n"
        
        if usage:
            content += f"--- USAGE ---\n{json.dumps(usage, indent=2)}\n\n"
        
        content += f"--- RAW RESPONSE ---\n{response_content}\n"
        
        if parsed_result is not None and success:
            content += f"\n--- PARSED RESULT ---\n{json.dumps(parsed_result, indent=2)}\n"
        
        # Write to file
        response_file = self.log_dir / f"response_{request_id}.txt"
        response_file.write_text(content, encoding="utf-8")
    
    def get_log_path(self) -> Path:
        """Return the log directory path."""
        return self.log_dir
