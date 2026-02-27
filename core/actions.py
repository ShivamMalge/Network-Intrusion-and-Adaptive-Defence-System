"""
Action Schema Phase 3.

Responsibility boundaries:
- Defines strict, immutable agent actions.
- Enforces structural validation rules for attacker and defender payloads.

Mutation constraints:
- Actions are strictly immutable Post-Initialization.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


class ActionType(Enum):
    # Attacker Actions
    SCAN = auto()
    EXPLOIT = auto()
    MOVE_LATERAL = auto()
    ATTACKER_NO_OP = auto()
    
    # Defender Actions
    PATCH = auto()
    ISOLATE = auto()
    DEPLOY_HONEYPOT = auto()
    RESET_PRIVILEGE = auto()
    DEFENDER_NO_OP = auto()


class InvalidActionError(Exception):
    pass


@dataclass(frozen=True)
class BaseAction:
    """
    Abstract baseline for Agent Actions.
    Must be treated as completely immutable.
    """
    agent_id: str
    action_type: ActionType
    target_node: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Structural validation stub to be overridden by subclasses."""
        pass


@dataclass(frozen=True)
class AttackerAction(BaseAction):
    """
    Immutable action payload emitted by an Attacker Agent.
    """

    def validate(self) -> None:
        requires_target = {
            ActionType.SCAN,
            ActionType.EXPLOIT,
            ActionType.MOVE_LATERAL
        }
        
        if self.action_type in requires_target and not self.target_node:
            raise InvalidActionError(f"Action '{self.action_type.name}' requires a target_node.")
            
        if self.action_type == ActionType.ATTACKER_NO_OP and self.target_node is not None:
            raise InvalidActionError(f"Action 'ATTACKER_NO_OP' must not specify a target_node.")


@dataclass(frozen=True)
class DefenderAction(BaseAction):
    """
    Immutable action payload emitted by a Defender Agent.
    """

    def validate(self) -> None:
        requires_target = {
            ActionType.PATCH,
            ActionType.ISOLATE,
            ActionType.RESET_PRIVILEGE,
            ActionType.DEPLOY_HONEYPOT
        }
        
        if self.action_type in requires_target and not self.target_node:
            raise InvalidActionError(f"Action '{self.action_type.name}' requires a target_node.")
            
        if self.action_type == ActionType.DEFENDER_NO_OP and self.target_node is not None:
            raise InvalidActionError(f"Action 'DEFENDER_NO_OP' must not specify a target_node.")


if __name__ == "__main__":
    print("--- Phase 3 self-test ---")

    # 1. Create a valid attacker action
    try:
        valid_act = AttackerAction(
            agent_id="attacker_1",
            action_type=ActionType.EXPLOIT,
            target_node="server_abc",
            metadata={"vuln_id": "ZD-999"}
        )
        print("Valid Action created successfully:")
        print(valid_act)
    except Exception as e:
        print(f"FAILED: valid action raised {e}")

    # 2. Create invalid action (must raise error)
    try:
        invalid_act = AttackerAction(
            agent_id="attacker_1",
            action_type=ActionType.SCAN,
            target_node=None  # Missing required target
        )
        print("FAILED: invalid action did not raise exception.")
    except InvalidActionError as e:
        print(f"\nCaught Expected InvalidActionError: {e}")

    try:
        invalid_def_act = DefenderAction(
            agent_id="def_1",
            action_type=ActionType.DEFENDER_NO_OP,
            target_node="server_abc" # Must have no target
        )
        print("FAILED: invalid defender action did not raise exception.")
    except InvalidActionError as e:
        print(f"Caught Expected InvalidActionError: {e}")

    # 3. Test Immutability
    print("\nTesting immutability constraints...")
    try:
        valid_act.target_node = 'mutated_node'
        print("FAILED: object is not frozen!")
    except Exception as e:
        print(f"Caught Expected Immutability Error: {type(e).__name__} - {e}")
