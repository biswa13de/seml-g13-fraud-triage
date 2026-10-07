"""Event consumer: subscribes to payment.decided and opens a case for every
BLOCK and every STEP_UP above the configured amount (FR5)."""
from common.config import settings
from common.events import consume_forever, redis_client_for_consumer
from common.logger import get_logger
from services.case_service.store import create_case, get_connection

logger = get_logger(__name__)

GROUP = "case-service"


def _should_case(event: dict) -> bool:
    if event["decision"] == "BLOCK":
        return True
    if event["decision"] == "STEP_UP" and event["amount"] >= settings.stepup_case_min_amount:
        return True
    return False


def run() -> None:
    client = redis_client_for_consumer(settings.redis_url)
    conn = get_connection()

    def handle(msg_id, event: dict) -> None:
        if not _should_case(event):
            return
        case_id = create_case(
            conn, txn_id=event["txn_id"], amount=event["amount"],
            risk_score=event["risk_score"], decision=event["decision"],
            reason_codes=event["reason_codes"],
        )
        if case_id:
            logger.info("Case opened", extra={"case_id": case_id, "txn_id": event["txn_id"]})

    consume_forever(client, settings.stream_payment_decided, GROUP, "case-consumer-1", handle)


if __name__ == "__main__":
    run()
