from datetime import datetime

from ..schemas.interaction import InteractionEvent
from .world import EnvironmentState


class EventLog:
    """
    Central event recorder for the simulation.

    Every observable interaction in an episode is recorded here.
    """

    def __init__(self, env: EnvironmentState):
        self.env = env

    def record(
        self,
        episode_id: str,
        actor: str,
        event_type: str,
        visibility: str,
        payload: dict,
    ) -> InteractionEvent:

        sequence_number = len(self.env.events) + 1

        event = InteractionEvent(
            event_id=f"EV_{sequence_number:06d}",
            episode_id=episode_id,
            timestamp=datetime.now(),
            sequence_number=sequence_number,
            actor=actor,
            event_type=event_type,
            visibility=visibility,
            payload=payload,
        )

        self.env.events.append(event)

        return event

    def get_events(
        self,
        episode_id: str,
    ) -> list[InteractionEvent]:

        return [
            event
            for event in self.env.events
            if event.episode_id == episode_id
        ]

    def get_customer_events(
        self,
        customer_id: str,
    ) -> list[InteractionEvent]:

        return [
            event
            for event in self.env.events
            if event.payload.get("customer_id")
            == customer_id
        ]

    def get_refund_events(
        self,
        refund_request_id: str,
    ) -> list[InteractionEvent]:

        return [
            event
            for event in self.env.events
            if event.payload.get(
                "refund_request_id"
            ) == refund_request_id
        ]