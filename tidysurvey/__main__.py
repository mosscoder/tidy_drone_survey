"""`python -m tidysurvey` — same entry point as the `tidysurvey` script
(what --detach re-executes, so it never depends on PATH)."""
from .cli import main

raise SystemExit(main())
