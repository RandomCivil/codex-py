# Serialized Conversation turns

A Conversation permits exactly one Active Conversation turn, and rejects concurrent appends as `conversation busy` rather than queueing them. Turns are durably created before their Agent runs start and retain their terminal result, so ordering and the history supplied to every later turn remain unambiguous across crashes and retries.

## Considered Options

- Execute or queue multiple turns for one Conversation — rejected because each new turn must receive a definite, complete preceding history.
- Record a turn only after Agent completion — rejected because a crash could leave executed work without its conversation linkage.
