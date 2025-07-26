import uuid

def create_mcp_message(sender: str, receiver: str, msg_type: str, payload: dict):
    return {
        "sender": sender,
        "receiver": receiver,
        "type": msg_type,
        "trace_id": str(uuid.uuid4()),
        "payload": payload
    }
