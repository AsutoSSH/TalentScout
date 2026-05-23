from langchain_core.messages import (
    messages_to_dict,
    messages_from_dict
)
from dotenv import load_dotenv
import os
from pymongo import MongoClient

# =========================================================
# MongoDB Setup
# =========================================================

load_dotenv()

MONGODB_URI = os.getenv(
    "MONGODB_URI"
)

MONGODB_DB_NAME = os.getenv(
    "MONGODB_DB_NAME"
)

MONGODB_COLLECTION_NAME = os.getenv(
    "MONGODB_COLLECTION_NAME"
)

mongo_client = MongoClient(
    MONGODB_URI
)

db = mongo_client[
    MONGODB_DB_NAME
]

sessions_collection = db[
    MONGODB_COLLECTION_NAME
]


# =========================================================
# Mongo Serialization Helpers
# =========================================================

def serialize_state(
    state: dict
):

    serialized = state.copy()

    # Convert LangChain messages
    serialized["messages"] = (
        messages_to_dict(
            state["messages"]
        )
    )

    return serialized


def deserialize_state(
    state: dict
):

    state["messages"] = (
        messages_from_dict(
            state["messages"]
        )
    )

    return state


# =========================================================
# Mongo Session Helpers
# =========================================================

def save_session(
    session_id: str,
    state: dict
):

    serialized_state = (
        serialize_state(state)
    )

    sessions_collection.update_one(

        {
            "session_id":
                session_id
        },

        {
            "$set": {
                "state":
                    serialized_state
            }
        },

        upsert=True
    )


def load_session(
    session_id: str
):

    doc = sessions_collection.find_one(
        {
            "session_id":
                session_id
        }
    )

    if not doc:

        raise ValueError(
            "Session not found"
        )

    return deserialize_state(
        doc["state"]
    )