# Folder access guide

WITS works in one folder, set by `folders.base` in `config.yaml`. Inside it are
three subfolders that WITS may always use.

## Where things go

- `inbox`: drop files here for WITS to read. This is the only folder WITS
  indexes by itself.
- `outbox`: WITS saves finished files here for you to pick up.
- `projects`: WITS saves working files for your projects here.

## When WITS needs another folder

WITS cannot open a folder outside those three on its own. It asks you. A
request shows up on the Inbox page as a card with:

- the folder it wants,
- what it wants to do there: "Read files in this folder", "Read and save files
  in this folder", or "Read and index files in this folder",
- the reason WITS gave.

You have three choices:

- **Approve 30 days**: WITS may use the folder for 30 days.
- **One day**: WITS may use the folder for 24 hours.
- **Deny**: nothing is granted.

## What a grant covers

A grant covers the folder and everything inside it, not just one file. If WITS
asks for a single file, the grant is for the folder that holds it. A grant ends
by itself when its time is up. Nothing renews it; WITS has to ask again.

## Seeing and revoking grants

The Inbox page has a section called "Folders WITS can use". It lists the three
base folders and every live grant, with the date it ends and how many days are
left. Press Revoke next to a grant to end it right away.

## What is never allowed

Secrets (keys, tokens, password stores) and system folders are refused outright,
and WITS cannot ask for them.

## Indexing

Nothing outside `inbox` is indexed automatically. A folder is indexed only if
you grant it with the "index" mode, or if you start a reindex yourself. A chat
message cannot start indexing of anything else.
