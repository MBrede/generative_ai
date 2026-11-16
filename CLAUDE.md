# Working conventions for this repo

## Git workflow

- Work happens on personal branches (`alwin`, `Max`), not directly on `main`. Switch to the
  relevant personal branch before making changes, and make sure it's up to date with `main`
  first (fast-forward if there are no unique commits on the personal branch yet).
- Split a larger piece of work into smaller commits with clear, short commit messages, when it
  makes sense to do so — e.g. separate a structural change (reordering/renaming sections) from
  the content changes that follow it. Don't force a split where the change is genuinely one
  coherent edit.
- Once a full piece of work (e.g. reworking one lecture session) is done, open a single pull
  request from the personal branch back into `main` covering the whole set of changes, rather
  than one PR per commit.

## Notes folder

- `notes/` is a local scratch space for discussion notes (e.g. content review/rework notes) and
  is gitignored — it is not meant to be committed or appear in PRs.
