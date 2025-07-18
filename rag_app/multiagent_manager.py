"""
Multi-Agent Manager for RAG Application

This module provides a multi-agent system that orchestrates multiple specialized
AI agents to work collaboratively on complex queries.
"""

import asyncio
import json
import uuid
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from openai import OpenAI
from .embedding_service import EmbeddingService
from .vector_store import VectorStore
from .mcp_tools import MCPToolsManager


class AgentType(Enum):
    """Types of specialized agents."""
    RESEARCH = "research"
    ANALYSIS = "analysis"
    WRITING = "writing"
    TOOL = "tool"
    RAG = "rag"
    COORDINATOR = "coordinator"


class CoordinationStrategy(Enum):
    """Strategies for coordinating multiple agents."""
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    COLLABORATIVE = "collaborative"
    ADAPTIVE = "adaptive"


@dataclass
class AgentResult:
    """Result from an individual agent."""
    agent_id: str
    agent_type: AgentType
    task: str
    result: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    success: bool = True
    error: Optional[str] = None
    tools_used: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class MultiAgentTask:
    """A task that can be processed by multiple agents."""
    task_id: str
    user_query: str
    agent_assignments: Dict[AgentType, str]  # agent_type -> specific task
    coordination_strategy: CoordinationStrategy
    priority: int = 1
    max_parallel_agents: int = 3
    timeout: float = 60.0
    context: Dict[str, Any] = field(default_factory=dict)


class SpecializedAgent:
    """Base class for specialized agents."""
    
    def __init__(
        self,
        agent_id: str,
        agent_type: AgentType,
        client: OpenAI,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.client = client
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = self._get_system_prompt()
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for this agent type."""
        prompts = {
            AgentType.RESEARCH: """You are a Research Agent specialized in gathering and synthesizing information.
Your role is to:
- Identify key topics and concepts in user queries
- Search for relevant information from available sources
- Collect comprehensive background information
- Identify knowledge gaps and areas requiring deeper investigation
- Provide well-sourced findings with citations

Focus on accuracy, comprehensiveness, and identifying the most relevant information.""",

            AgentType.ANALYSIS: """You are an Analysis Agent specialized in deep analysis and reasoning.
Your role is to:
- Analyze information and data provided by other agents
- Identify patterns, trends, and relationships
- Perform critical thinking and logical reasoning
- Compare and contrast different perspectives
- Draw insights and conclusions from available information
- Highlight potential biases or limitations in the analysis

Focus on thorough analysis, logical reasoning, and evidence-based conclusions.""",

            AgentType.WRITING: """You are a Writing Agent specialized in communication and presentation.
Your role is to:
- Synthesize information from multiple sources into coherent responses
- Adapt writing style to the intended audience and purpose
- Structure information clearly and logically
- Ensure proper grammar, clarity, and flow
- Create engaging and informative content
- Maintain consistency in tone and style

Focus on clear communication, proper structure, and engaging presentation.""",

            AgentType.TOOL: """You are a Tool Agent specialized in using external tools and services.
Your role is to:
- Execute tool calls to gather real-time information
- Perform file operations, code execution, and database queries
- Use web search and external APIs when needed
- Validate and process tool results
- Handle errors and retry mechanisms
- Provide structured summaries of tool outputs

Focus on efficient tool usage, error handling, and accurate result processing.""",

            AgentType.RAG: """You are a RAG Agent specialized in document retrieval and context-aware responses.
Your role is to:
- Search through document collections for relevant information
- Retrieve and rank documents by relevance
- Extract key information from retrieved documents
- Maintain context awareness across document sources
- Provide source attribution and citations
- Handle document-specific queries effectively

Focus on accurate retrieval, proper citations, and context-aware responses.""",

            AgentType.COORDINATOR: """You are a Coordinator Agent responsible for orchestrating multi-agent workflows.
Your role is to:
- Analyze user queries to determine required agent types
- Assign specific tasks to appropriate agents
- Coordinate communication between agents
- Synthesize results from multiple agents
- Ensure coherent and comprehensive final responses
- Handle conflicts and inconsistencies between agent outputs

Focus on effective coordination, conflict resolution, and comprehensive synthesis."""
        }
        return prompts.get(self.agent_type, "You are a helpful AI assistant.")
    
    async def process_async(self, task: str, context: Dict[str, Any] = None) -> AgentResult:
        """Process a task asynchronously."""
        start_time = datetime.now()
        
        try:
            # Prepare messages
            messages = [{"role": "system", "content": self.system_prompt}]
            
            # Add context if provided
            if context:
                context_str = f"Context from other agents:\n{json.dumps(context, indent=2)}"
                messages.append({"role": "user", "content": context_str})
            
            messages.append({"role": "user", "content": task})
            
            # Generate response
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            result_content = response.choices[0].message.content
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return AgentResult(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                task=task,
                result=result_content,
                execution_time=execution_time,
                success=True,
                metadata={
                    "model_used": self.model_name,
                    "tokens_used": response.usage.total_tokens if response.usage else 0
                }
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return AgentResult(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                task=task,
                result="",
                execution_time=execution_time,
                success=False,
                error=str(e)
            )
    
    async def process_async_stream(self, task: str, context: Dict[str, Any] = None):
        """Process a task asynchronously with streaming output."""
        start_time = datetime.now()
        
        try:
            # Prepare messages
            messages = [{"role": "system", "content": self.system_prompt}]
            
            # Add context if provided
            if context:
                context_str = f"Context from other agents:\n{json.dumps(context, indent=2)}"
                messages.append({"role": "user", "content": context_str})
            
            messages.append({"role": "user", "content": task})
            
            # Generate streaming response
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True
            )
            
            full_response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield {
                        "type": "content",
                        "agent_id": self.agent_id,
                        "agent_type": self.agent_type,
                        "content": content
                    }
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Yield final result
            yield {
                "type": "result",
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "result": AgentResult(
                    agent_id=self.agent_id,
                    agent_type=self.agent_type,
                    task=task,
                    result=full_response,
                    execution_time=execution_time,
                    success=True,
                    metadata={
                        "model_used": self.model_name,
                        "tokens_used": 0  # Not available in streaming mode
                    }
                )
            }
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            yield {
                "type": "result",
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "result": AgentResult(
                    agent_id=self.agent_id,
                    agent_type=self.agent_type,
                    task=task,
                    result="",
                    execution_time=execution_time,
                    success=False,
                    error=str(e)
                )
            }
    
    def process_sync(self, task: str, context: Dict[str, Any] = None) -> AgentResult:
        """Process a task synchronously."""
        # Run async method in sync context
        loop = None
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.process_async(task, context))


class ToolAgent(SpecializedAgent):
    """Specialized agent for tool usage."""
    
    def __init__(self, agent_id: str, client: OpenAI, model_name: str, 
                 mcp_tools_manager: MCPToolsManager, **kwargs):
        super().__init__(agent_id, AgentType.TOOL, client, model_name, **kwargs)
        self.mcp_tools_manager = mcp_tools_manager
    
    async def process_async(self, task: str, context: Dict[str, Any] = None) -> AgentResult:
        """Process task with tool usage capability."""
        start_time = datetime.now()
        
        try:
            # Get tools descriptions
            tools_descriptions = self.mcp_tools_manager.get_tool_descriptions()
            tools_description = "\n".join([f"- {name}: {desc}" for name, desc in tools_descriptions.items()])
            
            # Enhanced system prompt for tool usage
            enhanced_prompt = f"{self.system_prompt}\n\nAvailable tools:\n{tools_description}"
            
            messages = [{"role": "system", "content": enhanced_prompt}]
            
            if context:
                context_str = f"Context from other agents:\n{json.dumps(context, indent=2)}"
                messages.append({"role": "user", "content": context_str})
            
            messages.append({"role": "user", "content": task})
            
            # Get tools for function calling
            tools = self.mcp_tools_manager.get_tools_for_nutanix()
            
            # Generate response with tools
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                tools=tools,
                tool_choice="auto"
            )
            
            message_obj = response.choices[0].message
            assistant_message = message_obj.content or ""
            
            # Handle tool calls
            tools_used = []
            tool_results = []
            
            if message_obj.tool_calls:
                # Process tool calls
                tool_results = self.mcp_tools_manager.process_tool_calls(
                    [{"type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}} 
                     for tc in message_obj.tool_calls]
                )
                
                tools_used = [tc.function.name for tc in message_obj.tool_calls]
                
                # Add tool results to messages for follow-up
                for tool_call, tool_result in zip(message_obj.tool_calls, tool_results):
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{"id": tool_call.id, "type": "function", "function": {"name": tool_call.function.name, "arguments": tool_call.function.arguments}}]
                    })
                    messages.append({
                        "role": "tool",
                        "content": str(tool_result.result if tool_result.success else tool_result.error),
                        "tool_call_id": tool_call.id
                    })
                
                # Get final response after tool execution
                final_response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                assistant_message = final_response.choices[0].message.content
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return AgentResult(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                task=task,
                result=assistant_message,
                execution_time=execution_time,
                success=True,
                tools_used=tools_used,
                metadata={
                    "model_used": self.model_name,
                    "tools_called": len(tools_used),
                    "tool_results": [{"tool": tr.name, "success": tr.success} for tr in tool_results]
                }
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return AgentResult(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                task=task,
                result="",
                execution_time=execution_time,
                success=False,
                error=str(e)
            )
    
    async def process_async_stream(self, task: str, context: Dict[str, Any] = None):
        """Process task with tool usage capability and streaming output."""
        start_time = datetime.now()
        
        try:
            # Get tools descriptions
            tools_descriptions = self.mcp_tools_manager.get_tool_descriptions()
            tools_description = "\n".join([f"- {name}: {desc}" for name, desc in tools_descriptions.items()])
            
            # Enhanced system prompt for tool usage
            enhanced_prompt = f"{self.system_prompt}\n\nAvailable tools:\n{tools_description}"
            
            messages = [{"role": "system", "content": enhanced_prompt}]
            
            if context:
                context_str = f"Context from other agents:\n{json.dumps(context, indent=2)}"
                messages.append({"role": "user", "content": context_str})
            
            messages.append({"role": "user", "content": task})
            
            # Get tools for function calling
            tools = self.mcp_tools_manager.get_tools_for_nutanix()
            
            # Generate streaming response with tools
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                tools=tools,
                tool_choice="auto",
                stream=True
            )
            
            # Collect streaming response and tool calls
            tool_calls_dict = {}  # Track tool calls by index
            content_buffer = ""
            
            for chunk in stream:
                delta = chunk.choices[0].delta
                
                # Handle content
                if delta.content is not None:
                    content = delta.content
                    content_buffer += content
                    yield {
                        "type": "content",
                        "agent_id": self.agent_id,
                        "agent_type": self.agent_type,
                        "content": content
                    }
                
                # Handle tool calls
                if delta.tool_calls:
                    for tool_call_delta in delta.tool_calls:
                        index = tool_call_delta.index
                        if index not in tool_calls_dict:
                            tool_calls_dict[index] = {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""}
                            }
                        
                        if tool_call_delta.id:
                            tool_calls_dict[index]["id"] = tool_call_delta.id
                        
                        if tool_call_delta.function:
                            if tool_call_delta.function.name:
                                tool_calls_dict[index]["function"]["name"] += tool_call_delta.function.name
                            if tool_call_delta.function.arguments:
                                tool_calls_dict[index]["function"]["arguments"] += tool_call_delta.function.arguments
            
            # Handle tool calls
            tools_used = []
            tool_results = []
            assistant_message = content_buffer
            
            if tool_calls_dict:
                yield {
                    "type": "content",
                    "agent_id": self.agent_id,
                    "agent_type": self.agent_type,
                    "content": "\n\n🔧 **Using tools to enhance response...**\n\n"
                }
                
                # Convert tool calls dict to list
                tool_calls = [tool_calls_dict[i] for i in sorted(tool_calls_dict.keys())]
                
                # Process tool calls
                tool_results = self.mcp_tools_manager.process_tool_calls(tool_calls)
                tools_used = [tc["function"]["name"] for tc in tool_calls]
                
                # Show tool results
                formatted_results = self.mcp_tools_manager.format_tool_results_for_streaming(tool_results)
                yield {
                    "type": "content",
                    "agent_id": self.agent_id,
                    "agent_type": self.agent_type,
                    "content": formatted_results
                }
                
                # Add tool results to messages for follow-up
                messages.append({
                    "role": "assistant",
                    "content": content_buffer if content_buffer else None,
                    "tool_calls": tool_calls
                })
                
                for tool_call, tool_result in zip(tool_calls, tool_results):
                    messages.append({
                        "role": "tool",
                        "content": str(tool_result.result if tool_result.success else tool_result.error),
                        "tool_call_id": tool_call["id"]
                    })
                
                # Get final response after tool execution
                yield {
                    "type": "content",
                    "agent_id": self.agent_id,
                    "agent_type": self.agent_type,
                    "content": "\n**Final response:**\n"
                }
                
                final_stream = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=True
                )
                
                final_response = ""
                for chunk in final_stream:
                    if chunk.choices[0].delta.content is not None:
                        content = chunk.choices[0].delta.content
                        final_response += content
                        yield {
                            "type": "content",
                            "agent_id": self.agent_id,
                            "agent_type": self.agent_type,
                            "content": content
                        }
                
                assistant_message = final_response
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Yield final result
            yield {
                "type": "result",
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "result": AgentResult(
                    agent_id=self.agent_id,
                    agent_type=self.agent_type,
                    task=task,
                    result=assistant_message,
                    execution_time=execution_time,
                    success=True,
                    tools_used=tools_used,
                    metadata={
                        "model_used": self.model_name,
                        "tools_called": len(tools_used),
                        "tool_results": [{"tool": tr.name, "success": tr.success} for tr in tool_results]
                    }
                )
            }
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            yield {
                "type": "result",
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "result": AgentResult(
                    agent_id=self.agent_id,
                    agent_type=self.agent_type,
                    task=task,
                    result="",
                    execution_time=execution_time,
                    success=False,
                    error=str(e)
                )
            }


class RAGAgent(SpecializedAgent):
    """Specialized agent for RAG operations."""
    
    def __init__(self, agent_id: str, client: OpenAI, model_name: str,
                 embedding_service: EmbeddingService, vector_store: VectorStore, **kwargs):
        super().__init__(agent_id, AgentType.RAG, client, model_name, **kwargs)
        self.embedding_service = embedding_service
        self.vector_store = vector_store
    
    async def process_async(self, task: str, context: Dict[str, Any] = None) -> AgentResult:
        """Process task with RAG capability."""
        start_time = datetime.now()
        
        try:
            # Search for relevant documents
            retrieved_docs = self.vector_store.search_by_text(
                task, self.embedding_service, k=5
            )
            
            # Build context from retrieved documents
            context_parts = []
            for i, (doc, score) in enumerate(retrieved_docs):
                source = doc.metadata.get("source", "Unknown")
                context_parts.append(f"Document {i+1} (Source: {source}, Score: {score:.3f}):\n{doc.content}")
            
            context_text = "\n\n".join(context_parts)
            
            # Enhanced system prompt with retrieved context
            enhanced_prompt = f"{self.system_prompt}\n\nRetrieved Documents:\n{context_text}"
            
            messages = [{"role": "system", "content": enhanced_prompt}]
            
            if context:
                context_str = f"Context from other agents:\n{json.dumps(context, indent=2)}"
                messages.append({"role": "user", "content": context_str})
            
            messages.append({"role": "user", "content": task})
            
            # Generate response
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            result_content = response.choices[0].message.content
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return AgentResult(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                task=task,
                result=result_content,
                execution_time=execution_time,
                success=True,
                metadata={
                    "model_used": self.model_name,
                    "documents_retrieved": len(retrieved_docs),
                    "avg_relevance_score": sum(score for _, score in retrieved_docs) / len(retrieved_docs) if retrieved_docs else 0,
                    "sources": [doc.metadata.get("source", "Unknown") for doc, _ in retrieved_docs]
                }
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return AgentResult(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                task=task,
                result="",
                execution_time=execution_time,
                success=False,
                error=str(e)
            )
    
    async def process_async_stream(self, task: str, context: Dict[str, Any] = None):
        """Process task with RAG capability and streaming output."""
        start_time = datetime.now()
        
        try:
            # Search for relevant documents
            retrieved_docs = self.vector_store.search_by_text(
                task, self.embedding_service, k=5
            )
            
            # Build context from retrieved documents
            context_parts = []
            for i, (doc, score) in enumerate(retrieved_docs):
                source = doc.metadata.get("source", "Unknown")
                context_parts.append(f"Document {i+1} (Source: {source}, Score: {score:.3f}):\n{doc.content}")
            
            context_text = "\n\n".join(context_parts)
            
            # Enhanced system prompt with retrieved context
            enhanced_prompt = f"{self.system_prompt}\n\nRetrieved Documents:\n{context_text}"
            
            messages = [{"role": "system", "content": enhanced_prompt}]
            
            if context:
                context_str = f"Context from other agents:\n{json.dumps(context, indent=2)}"
                messages.append({"role": "user", "content": context_str})
            
            messages.append({"role": "user", "content": task})
            
            # Generate streaming response
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True
            )
            
            full_response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield {
                        "type": "content",
                        "agent_id": self.agent_id,
                        "agent_type": self.agent_type,
                        "content": content
                    }
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Yield final result
            yield {
                "type": "result",
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "result": AgentResult(
                    agent_id=self.agent_id,
                    agent_type=self.agent_type,
                    task=task,
                    result=full_response,
                    execution_time=execution_time,
                    success=True,
                    metadata={
                        "model_used": self.model_name,
                        "documents_retrieved": len(retrieved_docs),
                        "avg_relevance_score": sum(score for _, score in retrieved_docs) / len(retrieved_docs) if retrieved_docs else 0,
                        "sources": [doc.metadata.get("source", "Unknown") for doc, _ in retrieved_docs]
                    }
                )
            }
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            yield {
                "type": "result",
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "result": AgentResult(
                    agent_id=self.agent_id,
                    agent_type=self.agent_type,
                    task=task,
                    result="",
                    execution_time=execution_time,
                    success=False,
                    error=str(e)
                )
            }


class MultiAgentManager:
    """Manager for coordinating multiple AI agents."""
    
    def __init__(
        self,
        client: OpenAI,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store: Optional[VectorStore] = None,
        mcp_tools_manager: Optional[MCPToolsManager] = None,
        max_parallel_agents: int = 3
    ):
        self.client = client
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.mcp_tools_manager = mcp_tools_manager
        self.max_parallel_agents = max_parallel_agents
        
        # Initialize agents
        self.agents: Dict[str, SpecializedAgent] = {}
        self._initialize_agents()
        
        # Task tracking
        self.active_tasks: Dict[str, MultiAgentTask] = {}
        self.task_results: Dict[str, List[AgentResult]] = {}
    
    def _initialize_agents(self):
        """Initialize specialized agents."""
        base_kwargs = {
            "client": self.client,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        # Core agents
        self.agents["research_1"] = SpecializedAgent("research_1", AgentType.RESEARCH, **base_kwargs)
        self.agents["analysis_1"] = SpecializedAgent("analysis_1", AgentType.ANALYSIS, **base_kwargs)
        self.agents["writing_1"] = SpecializedAgent("writing_1", AgentType.WRITING, **base_kwargs)
        self.agents["coordinator_1"] = SpecializedAgent("coordinator_1", AgentType.COORDINATOR, **base_kwargs)
        
        # Optional agents based on available services
        if self.mcp_tools_manager:
            self.agents["tool_1"] = ToolAgent("tool_1", mcp_tools_manager=self.mcp_tools_manager, **base_kwargs)
        
        if self.embedding_service and self.vector_store:
            self.agents["rag_1"] = RAGAgent(
                "rag_1", 
                embedding_service=self.embedding_service,
                vector_store=self.vector_store,
                **base_kwargs
            )
    
    def analyze_query_complexity(self, query: str) -> Dict[str, Any]:
        """Analyze query to determine optimal agent assignment."""
        analysis = {
            "requires_research": any(keyword in query.lower() for keyword in ["what is", "explain", "describe", "tell me about"]),
            "requires_analysis": any(keyword in query.lower() for keyword in ["compare", "analyze", "evaluate", "pros and cons"]),
            "requires_tools": any(keyword in query.lower() for keyword in ["file", "database", "search", "execute", "run"]),
            "requires_documents": any(keyword in query.lower() for keyword in ["document", "source", "reference", "citation"]),
            "complexity_score": len(query.split()) / 10,  # Simple complexity heuristic
            "suggested_strategy": CoordinationStrategy.PARALLEL
        }
        
        # Determine coordination strategy
        if analysis["complexity_score"] > 2.0:
            analysis["suggested_strategy"] = CoordinationStrategy.COLLABORATIVE
        elif sum([analysis["requires_research"], analysis["requires_analysis"], 
                 analysis["requires_tools"], analysis["requires_documents"]]) > 2:
            analysis["suggested_strategy"] = CoordinationStrategy.SEQUENTIAL
        
        return analysis
    
    def create_task(
        self,
        user_query: str,
        coordination_strategy: Optional[CoordinationStrategy] = None,
        agent_assignments: Optional[Dict[AgentType, str]] = None
    ) -> MultiAgentTask:
        """Create a new multi-agent task."""
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        
        # Analyze query if no manual assignments
        if not agent_assignments:
            analysis = self.analyze_query_complexity(user_query)
            
            # Auto-assign agents based on analysis
            assignments = {}
            
            if analysis["requires_research"]:
                assignments[AgentType.RESEARCH] = f"Research the following query thoroughly: {user_query}"
            
            if analysis["requires_analysis"]:
                assignments[AgentType.ANALYSIS] = f"Analyze and provide insights on: {user_query}"
            
            if analysis["requires_tools"] and "tool_1" in self.agents:
                assignments[AgentType.TOOL] = f"Use appropriate tools to help answer: {user_query}"
            
            if analysis["requires_documents"] and "rag_1" in self.agents:
                assignments[AgentType.RAG] = f"Search documents for information about: {user_query}"
            
            # Always include writing agent for final synthesis
            assignments[AgentType.WRITING] = f"Synthesize information to answer: {user_query}"
            
            agent_assignments = assignments
            coordination_strategy = coordination_strategy or analysis["suggested_strategy"]
        
        return MultiAgentTask(
            task_id=task_id,
            user_query=user_query,
            agent_assignments=agent_assignments,
            coordination_strategy=coordination_strategy or CoordinationStrategy.PARALLEL,
            max_parallel_agents=self.max_parallel_agents
        )
    
    async def execute_task_async(self, task: MultiAgentTask, progress_callback: Optional[Callable] = None) -> List[AgentResult]:
        """Execute a multi-agent task asynchronously."""
        self.active_tasks[task.task_id] = task
        results = []
        
        try:
            if task.coordination_strategy == CoordinationStrategy.PARALLEL:
                results = await self._execute_parallel(task, progress_callback)
            elif task.coordination_strategy == CoordinationStrategy.SEQUENTIAL:
                results = await self._execute_sequential(task, progress_callback)
            elif task.coordination_strategy == CoordinationStrategy.COLLABORATIVE:
                results = await self._execute_collaborative(task, progress_callback)
            else:  # ADAPTIVE
                results = await self._execute_adaptive(task, progress_callback)
            
            self.task_results[task.task_id] = results
            return results
            
        finally:
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
    
    async def _execute_parallel(self, task: MultiAgentTask, progress_callback: Optional[Callable] = None) -> List[AgentResult]:
        """Execute agents in parallel."""
        if progress_callback:
            progress_callback("🚀 Starting parallel agent execution...")
        
        # Create tasks for all assigned agents
        agent_tasks = []
        for agent_type, agent_task in task.agent_assignments.items():
            # Find available agent of the required type
            agent = self._get_agent_by_type(agent_type)
            if agent:
                agent_tasks.append(agent.process_async(agent_task, task.context))
        
        # Execute all tasks in parallel
        if progress_callback:
            progress_callback(f"⚡ Running {len(agent_tasks)} agents in parallel...")
        
        results = await asyncio.gather(*agent_tasks, return_exceptions=True)
        
        # Filter out exceptions and convert to AgentResult objects
        valid_results = []
        for result in results:
            if isinstance(result, AgentResult):
                valid_results.append(result)
            elif isinstance(result, Exception):
                # Create error result
                error_result = AgentResult(
                    agent_id="unknown",
                    agent_type=AgentType.COORDINATOR,
                    task="parallel execution",
                    result="",
                    success=False,
                    error=str(result)
                )
                valid_results.append(error_result)
        
        if progress_callback:
            progress_callback(f"✅ Completed parallel execution with {len(valid_results)} results")
        
        return valid_results
    
    async def _execute_sequential(self, task: MultiAgentTask, progress_callback: Optional[Callable] = None) -> List[AgentResult]:
        """Execute agents sequentially, passing context between them."""
        if progress_callback:
            progress_callback("🔄 Starting sequential agent execution...")
        
        results = []
        shared_context = task.context.copy()
        
        # Define execution order
        execution_order = [AgentType.RESEARCH, AgentType.RAG, AgentType.TOOL, AgentType.ANALYSIS, AgentType.WRITING]
        
        for agent_type in execution_order:
            if agent_type in task.agent_assignments:
                agent = self._get_agent_by_type(agent_type)
                if agent:
                    if progress_callback:
                        progress_callback(f"🤖 Running {agent_type.value} agent...")
                    
                    result = await agent.process_async(task.agent_assignments[agent_type], shared_context)
                    results.append(result)
                    
                    # Add result to shared context for next agent
                    if result.success:
                        shared_context[f"{agent_type.value}_result"] = result.result
        
        if progress_callback:
            progress_callback(f"✅ Completed sequential execution with {len(results)} agents")
        
        return results
    
    async def _execute_collaborative(self, task: MultiAgentTask, progress_callback: Optional[Callable] = None) -> List[AgentResult]:
        """Execute agents collaboratively with multiple rounds of interaction."""
        if progress_callback:
            progress_callback("🤝 Starting collaborative agent execution...")
        
        all_results = []
        shared_context = task.context.copy()
        
        # Round 1: Initial parallel execution
        if progress_callback:
            progress_callback("📝 Round 1: Initial research and analysis...")
        
        round1_agents = [AgentType.RESEARCH, AgentType.RAG, AgentType.TOOL]
        round1_tasks = []
        
        for agent_type in round1_agents:
            if agent_type in task.agent_assignments:
                agent = self._get_agent_by_type(agent_type)
                if agent:
                    round1_tasks.append(agent.process_async(task.agent_assignments[agent_type], shared_context))
        
        round1_results = await asyncio.gather(*round1_tasks, return_exceptions=True)
        
        # Process round 1 results
        for result in round1_results:
            if isinstance(result, AgentResult):
                all_results.append(result)
                if result.success:
                    shared_context[f"{result.agent_type.value}_result"] = result.result
        
        # Round 2: Analysis based on round 1 results
        if AgentType.ANALYSIS in task.agent_assignments:
            if progress_callback:
                progress_callback("🔍 Round 2: Deep analysis of gathered information...")
            
            agent = self._get_agent_by_type(AgentType.ANALYSIS)
            if agent:
                analysis_result = await agent.process_async(task.agent_assignments[AgentType.ANALYSIS], shared_context)
                all_results.append(analysis_result)
                if analysis_result.success:
                    shared_context["analysis_result"] = analysis_result.result
        
        # Round 3: Final synthesis
        if AgentType.WRITING in task.agent_assignments:
            if progress_callback:
                progress_callback("✍️ Round 3: Final synthesis and writing...")
            
            agent = self._get_agent_by_type(AgentType.WRITING)
            if agent:
                writing_result = await agent.process_async(task.agent_assignments[AgentType.WRITING], shared_context)
                all_results.append(writing_result)
        
        if progress_callback:
            progress_callback(f"✅ Completed collaborative execution with {len(all_results)} total results")
        
        return all_results
    
    async def _execute_adaptive(self, task: MultiAgentTask, progress_callback: Optional[Callable] = None) -> List[AgentResult]:
        """Execute agents adaptively based on intermediate results."""
        if progress_callback:
            progress_callback("🧠 Starting adaptive agent execution...")
        
        # Start with research if available
        results = []
        shared_context = task.context.copy()
        
        # Adaptive logic: decide next steps based on query analysis and intermediate results
        analysis = self.analyze_query_complexity(task.user_query)
        
        if analysis["complexity_score"] > 1.5:
            # High complexity: use collaborative approach
            return await self._execute_collaborative(task, progress_callback)
        else:
            # Low complexity: use parallel approach
            return await self._execute_parallel(task, progress_callback)
    
    def _get_agent_by_type(self, agent_type: AgentType) -> Optional[SpecializedAgent]:
        """Get an available agent of the specified type."""
        for agent_id, agent in self.agents.items():
            if agent.agent_type == agent_type:
                return agent
        return None
    
    def _get_agent_icon(self, agent_type: AgentType) -> str:
        """Get icon for agent type."""
        icons = {
            AgentType.RESEARCH: "🔍",
            AgentType.ANALYSIS: "📊",
            AgentType.WRITING: "✍️",
            AgentType.TOOL: "🔧",
            AgentType.RAG: "📚",
            AgentType.COORDINATOR: "🎯"
        }
        return icons.get(agent_type, "🤖")
    
    def execute_task_sync(self, task: MultiAgentTask, progress_callback: Optional[Callable] = None) -> List[AgentResult]:
        """Execute a multi-agent task synchronously."""
        loop = None
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.execute_task_async(task, progress_callback))
    
    async def execute_task_stream(self, task: MultiAgentTask):
        """Execute a multi-agent task with streaming output."""
        self.active_tasks[task.task_id] = task
        
        try:
            if task.coordination_strategy == CoordinationStrategy.PARALLEL:
                async for update in self._execute_parallel_stream(task):
                    yield update
            elif task.coordination_strategy == CoordinationStrategy.SEQUENTIAL:
                async for update in self._execute_sequential_stream(task):
                    yield update
            elif task.coordination_strategy == CoordinationStrategy.COLLABORATIVE:
                async for update in self._execute_collaborative_stream(task):
                    yield update
            else:  # ADAPTIVE
                async for update in self._execute_adaptive_stream(task):
                    yield update
                    
        finally:
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
    
    async def _execute_parallel_stream(self, task: MultiAgentTask):
        """Execute agents in parallel with streaming output."""
        yield {
            "type": "progress",
            "message": "🚀 Starting parallel agent execution..."
        }
        
        # Start all agent streams
        agent_streams = []
        active_agents = []
        
        for agent_type, agent_task in task.agent_assignments.items():
            agent = self._get_agent_by_type(agent_type)
            if agent:
                active_agents.append((agent, agent_task, agent_type))
        
        yield {
            "type": "progress", 
            "message": f"⚡ Running {len(active_agents)} agents in parallel..."
        }
        
        # Create async generators for each agent
        for agent, agent_task, agent_type in active_agents:
            if hasattr(agent, 'process_async_stream'):
                agent_streams.append(agent.process_async_stream(agent_task, task.context))
            else:
                # Fallback for agents without streaming support
                agent_streams.append(self._fallback_stream(agent, agent_task, task.context))
        
        # Process streams concurrently
        agent_results = []
        completed_agents = [False] * len(agent_streams)
        
        while not all(completed_agents):
            for i, stream in enumerate(agent_streams):
                if completed_agents[i]:
                    continue
                    
                try:
                    update = await stream.__anext__()
                    
                    if update["type"] == "content":
                        # Add agent type indicator to content
                        agent_icon = self._get_agent_icon(update["agent_type"])
                        yield {
                            "type": "agent_content",
                            "agent_type": update["agent_type"],
                            "agent_id": update["agent_id"],
                            "content": update["content"],
                            "icon": agent_icon
                        }
                    elif update["type"] == "result":
                        agent_results.append(update["result"])
                        completed_agents[i] = True
                        
                        yield {
                            "type": "agent_complete",
                            "agent_type": update["agent_type"],
                            "agent_id": update["agent_id"],
                            "success": update["result"].success
                        }
                        
                except StopAsyncIteration:
                    completed_agents[i] = True
                except Exception as e:
                    completed_agents[i] = True
                    # Create error result
                    error_result = AgentResult(
                        agent_id=f"error_{i}",
                        agent_type=AgentType.COORDINATOR,
                        task="parallel execution",
                        result="",
                        success=False,
                        error=str(e)
                    )
                    agent_results.append(error_result)
        
        self.task_results[task.task_id] = agent_results
        
        yield {
            "type": "progress",
            "message": f"✅ Completed parallel execution with {len(agent_results)} results"
        }
        
        yield {
            "type": "complete",
            "results": agent_results
        }
    
    async def _execute_sequential_stream(self, task: MultiAgentTask):
        """Execute agents sequentially with streaming output."""
        yield {
            "type": "progress",
            "message": "🔄 Starting sequential agent execution..."
        }
        
        results = []
        shared_context = task.context.copy()
        
        # Define execution order
        execution_order = [AgentType.RESEARCH, AgentType.RAG, AgentType.TOOL, AgentType.ANALYSIS, AgentType.WRITING]
        
        for agent_type in execution_order:
            if agent_type in task.agent_assignments:
                agent = self._get_agent_by_type(agent_type)
                if agent:
                    yield {
                        "type": "progress",
                        "message": f"🤖 Running {agent_type.value} agent..."
                    }
                    
                    # Stream agent response
                    if hasattr(agent, 'process_async_stream'):
                        async for update in agent.process_async_stream(task.agent_assignments[agent_type], shared_context):
                            if update["type"] == "content":
                                agent_icon = self._get_agent_icon(update["agent_type"])
                                yield {
                                    "type": "agent_content",
                                    "agent_type": update["agent_type"],
                                    "agent_id": update["agent_id"], 
                                    "content": update["content"],
                                    "icon": agent_icon
                                }
                            elif update["type"] == "result":
                                result = update["result"]
                                results.append(result)
                                
                                # Add result to shared context for next agent
                                if result.success:
                                    shared_context[f"{agent_type.value}_result"] = result.result
                                
                                yield {
                                    "type": "agent_complete",
                                    "agent_type": update["agent_type"],
                                    "agent_id": update["agent_id"],
                                    "success": result.success
                                }
                    else:
                        # Fallback for non-streaming agents
                        result = await agent.process_async(task.agent_assignments[agent_type], shared_context)
                        results.append(result)
                        if result.success:
                            shared_context[f"{agent_type.value}_result"] = result.result
        
        self.task_results[task.task_id] = results
        
        yield {
            "type": "progress", 
            "message": f"✅ Completed sequential execution with {len(results)} agents"
        }
        
        yield {
            "type": "complete",
            "results": results
        }
    
    async def _execute_collaborative_stream(self, task: MultiAgentTask):
        """Execute agents collaboratively with streaming output."""
        yield {
            "type": "progress",
            "message": "🤝 Starting collaborative agent execution..."
        }
        
        all_results = []
        shared_context = task.context.copy()
        
        # Round 1: Initial parallel execution
        yield {
            "type": "progress",
            "message": "📝 Round 1: Initial research and analysis..."
        }
        
        round1_agents = [AgentType.RESEARCH, AgentType.RAG, AgentType.TOOL]
        round1_results = []
        
        for agent_type in round1_agents:
            if agent_type in task.agent_assignments:
                agent = self._get_agent_by_type(agent_type)
                if agent:
                    if hasattr(agent, 'process_async_stream'):
                        async for update in agent.process_async_stream(task.agent_assignments[agent_type], shared_context):
                            if update["type"] == "content":
                                agent_icon = self._get_agent_icon(update["agent_type"])
                                yield {
                                    "type": "agent_content",
                                    "agent_type": update["agent_type"],
                                    "agent_id": update["agent_id"],
                                    "content": update["content"],
                                    "icon": agent_icon
                                }
                            elif update["type"] == "result":
                                result = update["result"]
                                round1_results.append(result)
                                all_results.append(result)
                                if result.success:
                                    shared_context[f"{result.agent_type.value}_result"] = result.result
        
        # Round 2: Analysis based on round 1 results
        if AgentType.ANALYSIS in task.agent_assignments:
            yield {
                "type": "progress",
                "message": "🔍 Round 2: Deep analysis of gathered information..."
            }
            
            agent = self._get_agent_by_type(AgentType.ANALYSIS)
            if agent:
                if hasattr(agent, 'process_async_stream'):
                    async for update in agent.process_async_stream(task.agent_assignments[AgentType.ANALYSIS], shared_context):
                        if update["type"] == "content":
                            agent_icon = self._get_agent_icon(update["agent_type"])
                            yield {
                                "type": "agent_content",
                                "agent_type": update["agent_type"],
                                "agent_id": update["agent_id"],
                                "content": update["content"],
                                "icon": agent_icon
                            }
                        elif update["type"] == "result":
                            result = update["result"]
                            all_results.append(result)
                            if result.success:
                                shared_context["analysis_result"] = result.result
        
        # Round 3: Final synthesis
        if AgentType.WRITING in task.agent_assignments:
            yield {
                "type": "progress",
                "message": "✍️ Round 3: Final synthesis and writing..."
            }
            
            agent = self._get_agent_by_type(AgentType.WRITING)
            if agent:
                if hasattr(agent, 'process_async_stream'):
                    async for update in agent.process_async_stream(task.agent_assignments[AgentType.WRITING], shared_context):
                        if update["type"] == "content":
                            agent_icon = self._get_agent_icon(update["agent_type"])
                            yield {
                                "type": "agent_content",
                                "agent_type": update["agent_type"],
                                "agent_id": update["agent_id"],
                                "content": update["content"],
                                "icon": agent_icon
                            }
                        elif update["type"] == "result":
                            all_results.append(update["result"])
        
        self.task_results[task.task_id] = all_results
        
        yield {
            "type": "progress",
            "message": f"✅ Completed collaborative execution with {len(all_results)} total results"
        }
        
        yield {
            "type": "complete",
            "results": all_results
        }
    
    async def _execute_adaptive_stream(self, task: MultiAgentTask):
        """Execute agents adaptively with streaming output."""
        yield {
            "type": "progress",
            "message": "🧠 Starting adaptive agent execution..."
        }
        
        # Adaptive logic: decide next steps based on query analysis
        analysis = self.analyze_query_complexity(task.user_query)
        
        if analysis["complexity_score"] > 1.5:
            # High complexity: use collaborative approach
            async for update in self._execute_collaborative_stream(task):
                yield update
        else:
            # Low complexity: use parallel approach
            async for update in self._execute_parallel_stream(task):
                yield update
    
    async def _fallback_stream(self, agent: SpecializedAgent, task: str, context: Dict[str, Any]):
        """Fallback streaming for agents without native streaming support."""
        result = await agent.process_async(task, context)
        
        # Simulate streaming by yielding words
        words = result.result.split() if result.success else ["Error:", result.error]
        for word in words:
            yield {
                "type": "content",
                "agent_id": agent.agent_id,
                "agent_type": agent.agent_type,
                "content": word + " "
            }
        
        yield {
            "type": "result",
            "agent_id": agent.agent_id,
            "agent_type": agent.agent_type,
            "result": result
        }
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get statistics about available agents."""
        stats = {
            "total_agents": len(self.agents),
            "agents_by_type": {},
            "available_capabilities": []
        }
        
        for agent in self.agents.values():
            agent_type = agent.agent_type.value
            if agent_type not in stats["agents_by_type"]:
                stats["agents_by_type"][agent_type] = 0
            stats["agents_by_type"][agent_type] += 1
        
        # Available capabilities
        if "tool_1" in self.agents:
            stats["available_capabilities"].append("Tool Usage")
        if "rag_1" in self.agents:
            stats["available_capabilities"].append("Document Retrieval")
        
        stats["available_capabilities"].extend(["Research", "Analysis", "Writing", "Coordination"])
        
        return stats
    
    def synthesize_results(self, results: List[AgentResult], user_query: str) -> str:
        """Synthesize results from multiple agents into a final response."""
        if not results:
            return "No results were generated by the agents."
        
        # Separate successful and failed results
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        if not successful_results:
            return f"All agents failed to process the query. Errors: {'; '.join([r.error for r in failed_results])}"
        
        # Use writing agent for synthesis if available
        writing_agent = self._get_agent_by_type(AgentType.WRITING)
        if writing_agent:
            # Prepare context with all successful results
            synthesis_context = {
                "user_query": user_query,
                "agent_results": {
                    result.agent_type.value: result.result 
                    for result in successful_results
                }
            }
            
            synthesis_task = f"Synthesize the following agent results into a comprehensive answer for the user query: '{user_query}'"
            synthesis_result = writing_agent.process_sync(synthesis_task, synthesis_context)
            
            if synthesis_result.success:
                return synthesis_result.result
        
        # Fallback: simple concatenation with headers
        response_parts = [f"# Response to: {user_query}\n"]
        
        for result in successful_results:
            agent_name = result.agent_type.value.title()
            response_parts.append(f"## {agent_name} Agent:")
            response_parts.append(result.result)
            response_parts.append("")
        
        if failed_results:
            response_parts.append("## Errors:")
            for result in failed_results:
                response_parts.append(f"- {result.agent_type.value}: {result.error}")
        
        return "\n".join(response_parts) 