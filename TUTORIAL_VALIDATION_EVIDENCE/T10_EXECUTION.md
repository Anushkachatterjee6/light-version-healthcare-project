# T10 independent execution evidence

Executed against the running local backend on 2 November 2026.

| Request | Returned result |
|---|---|
| `POST /api/tutorials/10/run` with `{"operation":"classification"}` | `operation=classification`, held-out accuracy `1.0`, one confusion-matrix plot |
| `POST /api/tutorials/10/run` with `{"operation":"clustering","clusters":3}` | `operation=clustering`, `clusters=3`, cluster assignments present, two plots |

The values above come from the bundled synthetic teaching corpus. They are not clinical performance claims.
