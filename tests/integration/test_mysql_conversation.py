import asyncio
import os
import uuid

import pytest

from agent.conversation import ConversationBusyError, ConversationTurn, MySQLConversationStore
from agent.migration import _connection_pool, _parse_url, migrate_database


MYSQL_URL = os.environ.get("CODEX_TEST_MYSQL_URL")


@pytest.mark.integration
@pytest.mark.skipif(MYSQL_URL is None, reason="CODEX_TEST_MYSQL_URL is not configured; skipping MySQL integration test")
def test_mysql_conversation_append_serializes_active_turns():
    migrate_database(MYSQL_URL)

    async def exercise():
        async with _connection_pool(_parse_url(MYSQL_URL)) as pool:
            store = MySQLConversationStore(pool)
            conv_id = str(uuid.uuid4())
            first = ConversationTurn(conv_id, 1, str(uuid.uuid4()), "direct", "Hello", "running")
            await store.append(first)
            with pytest.raises(ConversationBusyError):
                await store.append(ConversationTurn(conv_id, 2, str(uuid.uuid4()), "direct", "Again", "pending"))
            await store.update(ConversationTurn(conv_id, 1, first.run_id, "direct", "Hello", "completed", "Hi"))
            conversation = await store.get(conv_id)
            assert [turn.sequence for turn in conversation.turns] == [1]
            assert conversation.turns[0].answer == "Hi"

    asyncio.run(exercise())


@pytest.mark.integration
@pytest.mark.skipif(MYSQL_URL is None, reason="CODEX_TEST_MYSQL_URL is not configured; skipping MySQL integration test")
def test_mysql_conversation_migration_is_idempotent_and_competing_appends_are_serialized():
    migrate_database(MYSQL_URL)
    migrate_database(MYSQL_URL)

    async def exercise():
        async with _connection_pool(_parse_url(MYSQL_URL)) as first_pool:
            async with _connection_pool(_parse_url(MYSQL_URL)) as second_pool:
                first_store = MySQLConversationStore(first_pool)
                second_store = MySQLConversationStore(second_pool)
                conv_id = str(uuid.uuid4())
                await first_store.append(
                    ConversationTurn(conv_id, 1, str(uuid.uuid4()), "direct", "First", "completed", "Done")
                )

                attempts = [
                    ConversationTurn(conv_id, 2, str(uuid.uuid4()), "direct", "Second A", "pending"),
                    ConversationTurn(conv_id, 2, str(uuid.uuid4()), "react", "Second B", "pending"),
                ]

                async def append(store, turn):
                    try:
                        await store.append(turn)
                    except ConversationBusyError as error:
                        return ("busy", error.sequence, error.run_id)
                    return ("appended", turn.run_id)

                results = await asyncio.gather(
                    append(first_store, attempts[0]),
                    append(second_store, attempts[1]),
                )

                assert sorted(result[0] for result in results) == ["appended", "busy"]
                conversation = await first_store.get(conv_id)
                assert [turn.sequence for turn in conversation.turns] == [1, 2]
                assert conversation.turns[0].status == "completed"

    asyncio.run(exercise())
