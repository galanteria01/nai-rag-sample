"""
MCP Tools Integration for RAG Application

This module provides MCP (Model Context Protocol) tools integration for enhanced
chat capabilities when RAG is disabled.
"""

import json
import os
import tempfile
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import requests
from dataclasses import dataclass


@dataclass
class ToolCall:
    """Represents a tool call made by the AI."""
    name: str
    arguments: Dict[str, Any]
    call_id: str
    timestamp: str


@dataclass
class ToolResult:
    """Represents the result of a tool call."""
    call_id: str
    name: str
    result: Any
    success: bool
    error: Optional[str] = None
    timestamp: Optional[str] = None


class MCPToolsManager:
    """
    Manager for MCP tools integration.
    Provides various tools that can be used to enhance chat capabilities.
    """
    
    def __init__(self, enabled_tools: Optional[List[str]] = None):
        """
        Initialize the MCP tools manager.
        
        Args:
            enabled_tools: List of tool names to enable. If None, all tools are enabled.
        """
        self.enabled_tools = enabled_tools or [
            "web_search",
            "runtime_logs",
            "runtime_errors",
            "file_operations",
            "code_execution",
            "memory_management"
        ]
        
        # Tool definitions for function calling
        self.tool_definitions = self._get_tool_definitions()
        
        # Runtime logs buffer
        self.runtime_logs = []
        self.runtime_errors = []
        
    def _get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Get tool definitions in OpenAI function calling format."""
        tools = []
        
        if "web_search" in self.enabled_tools:
            tools.append({
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for real-time information about any topic",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query to look up on the web"
                            }
                        },
                        "required": ["query"]
                    }
                }
            })
        
        if "runtime_logs" in self.enabled_tools:
            tools.append({
                "type": "function",
                "function": {
                    "name": "get_runtime_logs",
                    "description": "Get application runtime logs for debugging and monitoring",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filter_level": {
                                "type": "string",
                                "enum": ["debug", "info", "warning", "error"],
                                "description": "Filter logs by level"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of log entries to return",
                                "default": 50
                            }
                        }
                    }
                }
            })
        
        if "runtime_errors" in self.enabled_tools:
            tools.append({
                "type": "function",
                "function": {
                    "name": "get_runtime_errors",
                    "description": "Get application runtime errors for debugging",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of error entries to return",
                                "default": 20
                            }
                        }
                    }
                }
            })
        
        if "file_operations" in self.enabled_tools:
            tools.append({
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read the contents of a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the file to read"
                            },
                            "encoding": {
                                "type": "string",
                                "default": "utf-8",
                                "description": "File encoding"
                            }
                        },
                        "required": ["file_path"]
                    }
                }
            })
            
            tools.append({
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the file to write"
                            },
                            "content": {
                                "type": "string",
                                "description": "Content to write to the file"
                            },
                            "encoding": {
                                "type": "string",
                                "default": "utf-8",
                                "description": "File encoding"
                            }
                        },
                        "required": ["file_path", "content"]
                    }
                }
            })
        
        if "code_execution" in self.enabled_tools:
            tools.append({
                "type": "function",
                "function": {
                    "name": "execute_python",
                    "description": "Execute Python code and return the result",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Python code to execute"
                            },
                            "timeout": {
                                "type": "integer",
                                "default": 30,
                                "description": "Timeout in seconds"
                            }
                        },
                        "required": ["code"]
                    }
                }
            })
        
        if "memory_management" in self.enabled_tools:
            tools.append({
                "type": "function",
                "function": {
                    "name": "save_memory",
                    "description": "Save information to memory for future reference",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "key": {
                                "type": "string",
                                "description": "Memory key/identifier"
                            },
                            "value": {
                                "type": "string",
                                "description": "Information to save"
                            },
                            "category": {
                                "type": "string",
                                "description": "Category for organization"
                            }
                        },
                        "required": ["key", "value"]
                    }
                }
            })
            
            tools.append({
                "type": "function",
                "function": {
                    "name": "retrieve_memory",
                    "description": "Retrieve information from memory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "key": {
                                "type": "string",
                                "description": "Memory key to retrieve"
                            }
                        },
                        "required": ["key"]
                    }
                }
            })
        
        return tools
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return [tool["function"]["name"] for tool in self.tool_definitions]
    
    def get_tool_descriptions(self) -> Dict[str, str]:
        """Get descriptions of available tools."""
        return {
            tool["function"]["name"]: tool["function"]["description"]
            for tool in self.tool_definitions
        }
    
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        """
        Call a specific tool with given arguments.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Arguments for the tool
            
        Returns:
            ToolResult object with the result
        """
        call_id = f"{tool_name}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        timestamp = datetime.now().isoformat()
        
        try:
            if tool_name == "web_search":
                result = self._web_search(arguments.get("query", ""))
            elif tool_name == "get_runtime_logs":
                result = self._get_runtime_logs(
                    arguments.get("filter_level"),
                    arguments.get("limit", 50)
                )
            elif tool_name == "get_runtime_errors":
                result = self._get_runtime_errors(arguments.get("limit", 20))
            elif tool_name == "read_file":
                result = self._read_file(
                    arguments.get("file_path"),
                    arguments.get("encoding", "utf-8")
                )
            elif tool_name == "write_file":
                result = self._write_file(
                    arguments.get("file_path"),
                    arguments.get("content"),
                    arguments.get("encoding", "utf-8")
                )
            elif tool_name == "execute_python":
                result = self._execute_python(
                    arguments.get("code"),
                    arguments.get("timeout", 30)
                )
            elif tool_name == "save_memory":
                result = self._save_memory(
                    arguments.get("key"),
                    arguments.get("value"),
                    arguments.get("category")
                )
            elif tool_name == "retrieve_memory":
                result = self._retrieve_memory(arguments.get("key"))
            else:
                raise ValueError(f"Unknown tool: {tool_name}")
            
            return ToolResult(
                call_id=call_id,
                name=tool_name,
                result=result,
                success=True,
                timestamp=timestamp
            )
        
        except Exception as e:
            return ToolResult(
                call_id=call_id,
                name=tool_name,
                result=None,
                success=False,
                error=str(e),
                timestamp=timestamp
            )
    
    def _web_search(self, query: str) -> Dict[str, Any]:
        """Perform web search (placeholder - would need actual web search API)."""
        # This is a placeholder. In a real implementation, you'd use a web search API
        return {
            "query": query,
            "results": [
                {
                    "title": f"Search results for: {query}",
                    "snippet": "This is a placeholder for web search results. In a real implementation, this would use a web search API like Google Search API, Bing API, or similar.",
                    "url": "https://example.com"
                }
            ],
            "message": "Web search is not implemented. This is a placeholder response."
        }
    
    def _get_runtime_logs(self, filter_level: Optional[str], limit: int) -> List[Dict[str, Any]]:
        """Get runtime logs (placeholder - would integrate with actual logging)."""
        # This would integrate with actual application logging
        sample_logs = [
            {
                "timestamp": datetime.now().isoformat(),
                "level": "INFO",
                "message": "Application started successfully",
                "module": "main"
            },
            {
                "timestamp": datetime.now().isoformat(),
                "level": "DEBUG",
                "message": "RAG engine initialized",
                "module": "rag_engine"
            }
        ]
        
        if filter_level:
            sample_logs = [log for log in sample_logs if log["level"].lower() == filter_level.lower()]
        
        return sample_logs[:limit]
    
    def _get_runtime_errors(self, limit: int) -> List[Dict[str, Any]]:
        """Get runtime errors (placeholder - would integrate with actual error tracking)."""
        # This would integrate with actual error tracking
        return [
            {
                "timestamp": datetime.now().isoformat(),
                "error_type": "ValueError",
                "message": "No runtime errors detected",
                "module": "system"
            }
        ][:limit]
    
    def _read_file(self, file_path: str, encoding: str) -> str:
        """Read file contents."""
        if not file_path:
            raise ValueError("File path is required")
        
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except Exception as e:
            raise RuntimeError(f"Failed to read file {file_path}: {str(e)}")
    
    def _write_file(self, file_path: str, content: str, encoding: str) -> str:
        """Write content to file."""
        if not file_path or content is None:
            raise ValueError("File path and content are required")
        
        try:
            with open(file_path, 'w', encoding=encoding) as f:
                f.write(content)
            return f"Successfully wrote {len(content)} characters to {file_path}"
        except Exception as e:
            raise RuntimeError(f"Failed to write file {file_path}: {str(e)}")
    
    def _execute_python(self, code: str, timeout: int) -> Dict[str, Any]:
        """Execute Python code (placeholder - would need secure execution environment)."""
        # This is a placeholder. In a real implementation, you'd use a secure execution environment
        return {
            "code": code,
            "result": "Code execution is not implemented for security reasons. This is a placeholder response.",
            "stdout": "",
            "stderr": "",
            "execution_time": 0
        }
    
    def _save_memory(self, key: str, value: str, category: Optional[str]) -> str:
        """Save to memory (placeholder - would integrate with persistent storage)."""
        # This would integrate with persistent storage
        return f"Memory saved: {key} = {value[:50]}{'...' if len(value) > 50 else ''}"
    
    def _retrieve_memory(self, key: str) -> str:
        """Retrieve from memory (placeholder - would integrate with persistent storage)."""
        # This would integrate with persistent storage
        return f"Memory retrieval for key '{key}' is not implemented. This is a placeholder response."
    
    def process_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[ToolResult]:
        """
        Process multiple tool calls.
        
        Args:
            tool_calls: List of tool call dictionaries
            
        Returns:
            List of ToolResult objects
        """
        results = []
        
        for tool_call in tool_calls:
            if tool_call.get("type") == "function":
                function_call = tool_call.get("function", {})
                tool_name = function_call.get("name")
                arguments = json.loads(function_call.get("arguments", "{}"))
                
                result = self.call_tool(tool_name, arguments)
                results.append(result)
        
        return results
    
    def format_tool_results_for_chat(self, tool_results: List[ToolResult]) -> str:
        """
        Format tool results for display in chat with accordion-style layout.
        
        Args:
            tool_results: List of tool results
            
        Returns:
            Formatted string with accordion-style display for chat
        """
        if not tool_results:
            return ""
        
        # Group results by success/failure
        successful_results = [r for r in tool_results if r.success]
        failed_results = [r for r in tool_results if not r.success]
        
        formatted_parts = []
        
        # Add header section
        tool_count = len(tool_results)
        success_count = len(successful_results)
        
        if tool_count > 0:
            formatted_parts.append(f"## 🔧 Tool Results ({success_count}/{tool_count} successful)")
            formatted_parts.append("")
        
        # Format successful results with accordion-style collapsible sections
        for i, result in enumerate(successful_results, 1):
            tool_icon = self._get_tool_icon(result.name)
            formatted_parts.append(f"### {tool_icon} {result.name}")
            
            # Add timestamp if available
            if result.timestamp:
                formatted_parts.append(f"*Executed at: {result.timestamp}*")
            
            # Format the result based on its type
            formatted_result = self._format_tool_result_content(result)
            formatted_parts.append(formatted_result)
            formatted_parts.append("")
        
        # Format failed results
        for i, result in enumerate(failed_results, 1):
            tool_icon = self._get_tool_icon(result.name)
            formatted_parts.append(f"### ❌ {tool_icon} {result.name} - Failed")
            
            if result.timestamp:
                formatted_parts.append(f"*Failed at: {result.timestamp}*")
            
            formatted_parts.append(f"**Error:** {result.error}")
            formatted_parts.append("")
        
        return "\n".join(formatted_parts)
    
    def _get_tool_icon(self, tool_name: str) -> str:
        """Get appropriate icon for tool name."""
        icon_map = {
            "web_search": "🌐",
            "get_runtime_logs": "📊",
            "get_runtime_errors": "🐛",
            "read_file": "📄",
            "write_file": "📝",
            "execute_python": "💻",
            "save_memory": "🧠",
            "retrieve_memory": "🧠"
        }
        return icon_map.get(tool_name, "🔧")
    
    def _format_tool_result_content(self, result: ToolResult) -> str:
        """Format tool result content based on tool type."""
        if result.name == "web_search":
            return self._format_web_search_result(result.result)
        elif result.name in ["get_runtime_logs", "get_runtime_errors"]:
            return self._format_log_result(result.result)
        elif result.name in ["read_file", "write_file"]:
            return self._format_file_result(result.result, result.name)
        elif result.name == "execute_python":
            return self._format_code_execution_result(result.result)
        elif result.name in ["save_memory", "retrieve_memory"]:
            return self._format_memory_result(result.result)
        else:
            return self._format_generic_result(result.result)
    
    def _format_web_search_result(self, result: Any) -> str:
        """Format web search results."""
        if isinstance(result, dict):
            formatted = [f"**Query:** {result.get('query', 'N/A')}"]
            
            if 'results' in result:
                formatted.append("**Results:**")
                for i, res in enumerate(result['results'][:3], 1):  # Show top 3 results
                    formatted.append(f"{i}. **{res.get('title', 'No title')}**")
                    formatted.append(f"   {res.get('snippet', 'No snippet available')}")
                    if res.get('url'):
                        formatted.append(f"   🔗 {res['url']}")
            
            if 'message' in result:
                formatted.append(f"\n*Note: {result['message']}*")
            
            return "\n".join(formatted)
        
        return f"```\n{str(result)}\n```"
    
    def _format_log_result(self, result: Any) -> str:
        """Format log/error results."""
        if isinstance(result, list):
            if not result:
                return "No logs found."
            
            formatted = ["**Log Entries:**"]
            for i, log in enumerate(result[:5], 1):  # Show top 5 logs
                if isinstance(log, dict):
                    timestamp = log.get('timestamp', 'N/A')
                    level = log.get('level', 'INFO')
                    message = log.get('message', 'No message')
                    module = log.get('module', 'Unknown')
                    
                    formatted.append(f"{i}. `[{timestamp}] {level}` - {message}")
                    formatted.append(f"   *Module: {module}*")
                else:
                    formatted.append(f"{i}. {str(log)}")
            
            if len(result) > 5:
                formatted.append(f"\n*... and {len(result) - 5} more entries*")
            
            return "\n".join(formatted)
        
        return f"```\n{str(result)}\n```"
    
    def _format_file_result(self, result: Any, tool_name: str) -> str:
        """Format file operation results."""
        if tool_name == "read_file":
            if isinstance(result, str):
                # Show preview of file content
                lines = result.split('\n')
                if len(lines) > 10:
                    preview = '\n'.join(lines[:10])
                    return f"**File Content Preview:**\n```\n{preview}\n... ({len(lines)} total lines)\n```"
                else:
                    return f"**File Content:**\n```\n{result}\n```"
        elif tool_name == "write_file":
            return f"**Result:** {result}"
        
        return f"```\n{str(result)}\n```"
    
    def _format_code_execution_result(self, result: Any) -> str:
        """Format code execution results."""
        if isinstance(result, dict):
            formatted = []
            
            if 'code' in result:
                formatted.append("**Code Executed:**")
                formatted.append(f"```python\n{result['code']}\n```")
            
            if 'result' in result:
                formatted.append("**Result:**")
                formatted.append(f"```\n{result['result']}\n```")
            
            if 'stdout' in result and result['stdout']:
                formatted.append("**Output:**")
                formatted.append(f"```\n{result['stdout']}\n```")
            
            if 'stderr' in result and result['stderr']:
                formatted.append("**Errors:**")
                formatted.append(f"```\n{result['stderr']}\n```")
            
            if 'execution_time' in result:
                formatted.append(f"**Execution Time:** {result['execution_time']}s")
            
            return "\n".join(formatted)
        
        return f"```\n{str(result)}\n```"
    
    def _format_memory_result(self, result: Any) -> str:
        """Format memory operation results."""
        return f"**Result:** {result}"
    
    def _format_generic_result(self, result: Any) -> str:
        """Format generic tool results."""
        if isinstance(result, dict):
            formatted = ["**Result:**"]
            for key, value in result.items():
                formatted.append(f"- **{key}:** {value}")
            return "\n".join(formatted)
        elif isinstance(result, list):
            if not result:
                return "No results found."
            formatted = ["**Results:**"]
            for i, item in enumerate(result[:5], 1):
                formatted.append(f"{i}. {item}")
            if len(result) > 5:
                formatted.append(f"... and {len(result) - 5} more items")
            return "\n".join(formatted)
        else:
            return f"**Result:** {result}"
    
    def format_tool_results_for_streaming(self, tool_results: List[ToolResult]) -> str:
        """
        Format tool results for streaming display with enhanced accordion-style layout.
        This method is optimized for streaming contexts where results appear progressively.
        
        Args:
            tool_results: List of tool results
            
        Returns:
            Formatted string for streaming display
        """
        if not tool_results:
            return ""
        
        # Start with a clean section header
        formatted_parts = ["", "---", ""]
        
        # Add main header
        successful_count = sum(1 for r in tool_results if r.success)
        total_count = len(tool_results)
        
        if successful_count == total_count:
            formatted_parts.append(f"## ✅ Tool Results ({total_count} successful)")
        else:
            formatted_parts.append(f"## 🔧 Tool Results ({successful_count}/{total_count} successful)")
        
        formatted_parts.append("")
        
        # Process each tool result
        for result in tool_results:
            if result.success:
                tool_icon = self._get_tool_icon(result.name)
                formatted_parts.append(f"### {tool_icon} {result.name}")
                
                # Add a brief description of what the tool does
                tool_description = self._get_tool_description(result.name)
                if tool_description:
                    formatted_parts.append(f"*{tool_description}*")
                
                # Format the actual result
                formatted_result = self._format_tool_result_content(result)
                formatted_parts.append(formatted_result)
                formatted_parts.append("")
            else:
                # Format failed results
                tool_icon = self._get_tool_icon(result.name)
                formatted_parts.append(f"### ❌ {tool_icon} {result.name} - Failed")
                formatted_parts.append(f"**Error:** {result.error}")
                formatted_parts.append("")
        
        formatted_parts.append("---")
        formatted_parts.append("")
        
        return "\n".join(formatted_parts)
    
    def _get_tool_description(self, tool_name: str) -> str:
        """Get a brief description of what the tool does."""
        descriptions = {
            "web_search": "Searched the web for real-time information",
            "get_runtime_logs": "Retrieved application runtime logs",
            "get_runtime_errors": "Checked for runtime errors",
            "read_file": "Read file contents",
            "write_file": "Wrote content to file",
            "execute_python": "Executed Python code",
            "save_memory": "Saved information to memory",
            "retrieve_memory": "Retrieved information from memory"
        }
        return descriptions.get(tool_name, "")
    
    def get_tools_for_openai(self) -> List[Dict[str, Any]]:
        """Get tools formatted for OpenAI function calling."""
        return self.tool_definitions 