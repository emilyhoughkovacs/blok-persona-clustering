"""
Persona-based behavioral simulation agents.

This module provides classes for instantiating Claude-powered agents
that simulate customer behavior based on empirically-derived personas.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


@dataclass
class PersonaAgent:
    """
    Wraps the Claude API with a persona's system prompt.

    Each agent is initialized with behavioral parameters derived from
    customer clustering, enabling consistent persona-based responses.

    Attributes:
        cluster_id: The cluster this persona represents
        persona_name: Human-readable persona name
        system_prompt: The system prompt defining agent behavior
        client: Anthropic client instance (None in mock mode)
        mock_mode: If True, returns templated responses without API calls
    """

    cluster_id: int
    persona_name: str
    system_prompt: str
    client: Any = None
    mock_mode: bool = False
    model: str = "claude-sonnet-4-20250514"

    @classmethod
    def from_persona_data(
        cls,
        persona_data: dict,
        client: Any = None,
        mock_mode: bool = False,
        model: str = "claude-sonnet-4-20250514"
    ) -> PersonaAgent:
        """Create a PersonaAgent from persona dictionary."""
        return cls(
            cluster_id=persona_data.get("cluster_id", -1),
            persona_name=persona_data["persona_name"],
            system_prompt=persona_data["agent_system_prompt"],
            client=client,
            mock_mode=mock_mode,
            model=model
        )

    def respond(self, scenario: str, max_tokens: int = 500) -> str:
        """
        Generate a response to a product/purchase scenario.

        Args:
            scenario: The scenario or question to respond to
            max_tokens: Maximum response length

        Returns:
            The agent's response as this persona
        """
        if self.mock_mode:
            return self._mock_response(scenario)

        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "anthropic package not installed. "
                "Run: pip install anthropic"
            )

        if self.client is None:
            raise ValueError(
                "No Anthropic client provided. Either pass a client or use mock_mode=True"
            )

        message = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=self.system_prompt,
            messages=[{"role": "user", "content": scenario}]
        )

        return message.content[0].text

    def respond_with_decision(
        self,
        scenario: str,
        max_tokens: int = 500
    ) -> dict:
        """
        Generate a structured response with explicit decision and reasoning.

        Args:
            scenario: The scenario or question to respond to
            max_tokens: Maximum response length

        Returns:
            Dictionary with 'decision', 'reasoning', and 'raw_response'
        """
        structured_prompt = f"""{scenario}

Please respond with:
1. DECISION: [Yes/No/Maybe] - Would you make this purchase?
2. REASONING: Brief explanation of your decision (2-3 sentences)
3. KEY FACTORS: What were the most important factors in your decision?"""

        raw_response = self.respond(structured_prompt, max_tokens)

        # Parse the response (basic extraction)
        decision = self._extract_decision(raw_response)

        return {
            "persona_name": self.persona_name,
            "cluster_id": self.cluster_id,
            "decision": decision,
            "raw_response": raw_response
        }

    def _extract_decision(self, response: str) -> str:
        """Extract decision from structured response."""
        import re
        response_lower = response.lower()

        # Look for explicit decision markers (handles markdown bold, brackets, etc.)
        # Matches: "DECISION: Yes", "**DECISION:** No", "DECISION: [Maybe]", etc.
        decision_pattern = r'\*{0,2}decision\*{0,2}[:\s]+\[?(\w+)\]?'
        match = re.search(decision_pattern, response_lower)

        if match:
            decision = match.group(1)
            if decision in ('yes', 'y'):
                return "Yes"
            elif decision in ('no', 'n'):
                return "No"
            elif decision in ('maybe', 'uncertain', 'unsure'):
                return "Maybe"

        # Fallback: look for keywords in first 200 chars
        first_part = response_lower[:200]
        if "i would buy" in first_part or "i'll take" in first_part or "yes," in first_part:
            return "Yes"
        elif "i would not" in first_part or "i wouldn't" in first_part or "no," in first_part:
            return "No"

        return "Unclear"

    def _mock_response(self, scenario: str) -> str:
        """Generate a mock response for testing without API."""
        # Persona-specific mock responses based on key traits
        mock_templates = {
            "Mainstream Shopper": (
                "As a typical weekday shopper, I'd consider this purchase carefully. "
                "I usually buy what I need and move on. Given this scenario, I'd likely "
                "proceed if it meets my specific need and the price is reasonable."
            ),
            "Weekend Buyer": (
                "I typically browse on weekends when I have time. This seems interesting, "
                "but I'd want to think it over during my weekend shopping time."
            ),
            "Cash Customer": (
                "I prefer to pay upfront with boleto. If this requires installments or "
                "credit, I'd hesitate. I don't like carrying debt for purchases."
            ),
            "High-Value Financing Shopper": (
                "I'm comfortable with larger purchases when I can spread payments. "
                "If 10x installments are available, the monthly cost matters more than total price."
            ),
            "Bulk Buyer": (
                "I prefer to bundle purchases together. If there's a deal for buying multiple, "
                "I'd be more interested. Single items feel less efficient to me."
            ),
            "Loyal Explorer Customer": (
                "I'm always open to trying new categories. As a repeat customer, I trust this "
                "marketplace and would consider exploring this option."
            ),
            "Critical Shopper": (
                "I have high standards. Before deciding, I'd want to see the reviews carefully. "
                "If there are quality concerns, I'd pass regardless of the price."
            )
        }

        base_response = mock_templates.get(
            self.persona_name,
            f"[Mock response for {self.persona_name}] Considering the scenario..."
        )

        return f"[MOCK MODE] {base_response}\n\nScenario received: {scenario[:100]}..."


class PersonaSimulator:
    """
    Runs scenarios across all personas and collects responses.

    This class manages loading personas, instantiating agents, and
    running batch simulations across multiple scenarios.
    """

    def __init__(
        self,
        personas_path: str | Path = None,
        mock_mode: bool = False,
        model: str = "claude-sonnet-4-20250514"
    ):
        """
        Initialize the simulator.

        Args:
            personas_path: Path to personas.json. If None, uses default location.
            mock_mode: If True, all agents use mock responses
            model: Claude model to use for API calls
        """
        if personas_path is None:
            # Default path relative to this file
            personas_path = Path(__file__).parent.parent / "data/processed/personas.json"

        self.personas_path = Path(personas_path)
        self.mock_mode = mock_mode
        self.model = model
        self.personas_data: dict = {}
        self.agents: dict[int, PersonaAgent] = {}
        self._client = None

    def load_personas(self) -> dict:
        """Load personas from JSON file."""
        with open(self.personas_path, "r") as f:
            data = json.load(f)

        self.personas_data = data
        return data

    def _get_client(self):
        """Get or create Anthropic client."""
        if self.mock_mode:
            return None

        if self._client is None:
            if not ANTHROPIC_AVAILABLE:
                raise ImportError(
                    "anthropic package not installed. "
                    "Run: pip install anthropic"
                )

            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY not found in environment. "
                    "Set it or use mock_mode=True"
                )

            self._client = anthropic.Anthropic(api_key=api_key)

        return self._client

    def initialize_agents(self) -> dict[int, PersonaAgent]:
        """
        Create PersonaAgent instances for all personas.

        Returns:
            Dictionary mapping cluster_id to PersonaAgent
        """
        if not self.personas_data:
            self.load_personas()

        client = self._get_client()

        for cluster_id_str, persona_data in self.personas_data["personas"].items():
            cluster_id = int(cluster_id_str)
            persona_data["cluster_id"] = cluster_id

            self.agents[cluster_id] = PersonaAgent.from_persona_data(
                persona_data,
                client=client,
                mock_mode=self.mock_mode,
                model=self.model
            )

        return self.agents

    def run_scenario(self, scenario: str, structured: bool = True) -> pd.DataFrame:
        """
        Run a single scenario across all personas.

        Args:
            scenario: The scenario text to present to each persona
            structured: If True, use respond_with_decision for parsed output

        Returns:
            DataFrame with columns: cluster_id, persona_name, decision, response
        """
        if not self.agents:
            self.initialize_agents()

        results = []

        for cluster_id, agent in sorted(self.agents.items()):
            if structured:
                result = agent.respond_with_decision(scenario)
                results.append({
                    "cluster_id": cluster_id,
                    "persona_name": result["persona_name"],
                    "decision": result["decision"],
                    "response": result["raw_response"]
                })
            else:
                response = agent.respond(scenario)
                results.append({
                    "cluster_id": cluster_id,
                    "persona_name": agent.persona_name,
                    "decision": None,
                    "response": response
                })

        return pd.DataFrame(results)

    def run_batch(
        self,
        scenarios: list[dict],
        structured: bool = True
    ) -> pd.DataFrame:
        """
        Run multiple scenarios across all personas.

        Args:
            scenarios: List of dicts with 'name' and 'text' keys
            structured: If True, use respond_with_decision for parsed output

        Returns:
            DataFrame with columns: scenario_name, cluster_id, persona_name, decision, response
        """
        all_results = []

        for scenario in scenarios:
            scenario_name = scenario.get("name", "unnamed")
            scenario_text = scenario["text"]

            df = self.run_scenario(scenario_text, structured=structured)
            df["scenario_name"] = scenario_name
            all_results.append(df)

        return pd.concat(all_results, ignore_index=True)

    def get_persona_summary(self) -> pd.DataFrame:
        """Get a summary of all loaded personas."""
        if not self.personas_data:
            self.load_personas()

        summaries = []
        for cluster_id_str, persona in self.personas_data["personas"].items():
            summaries.append({
                "cluster_id": int(cluster_id_str),
                "persona_name": persona["persona_name"],
                "size": persona["size"],
                "percentage": f"{persona['percentage']:.1f}%"
            })

        return pd.DataFrame(summaries).sort_values("cluster_id")
