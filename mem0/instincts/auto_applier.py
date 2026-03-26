"""
Auto-applier for instincts.

Intercepts operations and applies matching instincts automatically.
"""

from typing import Any, Dict, List, Optional

from .instincts import Instinct, InstinctRegistry


class InstinctApplier:
    """
    Automatically applies instincts to matching operations.

    ECC-style auto-apply:
    - Check operation context against instincts
    - Apply matching instinct actions
    - Track effectiveness
    """

    def __init__(self, registry: InstinctRegistry):
        """
        Initialize applier.

        Args:
            registry: InstinctRegistry to pull instincts from
        """
        self.registry = registry

    async def apply_to_operation(
        self,
        method_name: str,
        args: tuple,
        kwargs: dict,
        project_id: str = "global",
    ) -> tuple[tuple, dict]:
        """
        Apply instincts to an operation.

        Args:
            method_name: Name of the method being called
            args: Original positional arguments
            kwargs: Original keyword arguments
            project_id: Project identifier

        Returns:
            Tuple of (modified_args, modified_kwargs)
        """
        # Get applicable instincts
        instincts = self.registry.get_instincts(project_id)
        instincts.extend(self.registry.get_global_instincts())

        modified_args = args
        modified_kwargs = kwargs

        # Apply each matching instinct
        for instinct in instincts:
            if not instinct.enabled:
                continue

            context = {"method": method_name, "args": args, "kwargs": kwargs}

            # Check if instinct triggers
            if self._should_trigger(instinct, context):
                # Apply the instinct
                modified_args, modified_kwargs = self._apply_instinct(
                    instinct,
                    modified_args,
                    modified_kwargs,
                )

        return modified_args, modified_kwargs

    def _should_trigger(self, instinct: Instinct, context: Dict[str, Any]) -> bool:
        """Check if instinct should trigger on context."""
        trigger = instinct.trigger

        if trigger.action_type == "method":
            return context.get("method") == trigger.content
        elif trigger.action_type == "pattern":
            pattern = trigger.content
            prompt = str(context.get("kwargs", {}))
            return pattern in prompt

        return False

    def _apply_instinct(
        self,
        instinct: Instinct,
        args: tuple,
        kwargs: dict,
    ) -> tuple[tuple, dict]:
        """Apply instinct action to args/kwargs."""
        action = instinct.action

        if action.action_type == "modify_args":
            if isinstance(action.content, dict):
                modified_kwargs = {**kwargs, **action.content}
                return args, modified_kwargs

        return args, kwargs

    async def record_result(
        self,
        instinct_id: str,
        successful: bool,
    ) -> None:
        """Record instinct application result."""
        await self.registry.record_application(instinct_id, successful)
