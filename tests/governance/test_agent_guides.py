# Copyright Xingyu Chen.
# Keeps the two agent guides byte-identical.

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class AgentGuideTests(unittest.TestCase):
    def test_claude_and_agents_guides_are_byte_identical(self) -> None:
        claude = (ROOT / "CLAUDE.md").read_bytes()
        agents = (ROOT / "AGENTS.md").read_bytes()
        self.assertEqual(claude, agents, "CLAUDE.md and AGENTS.md must stay byte-identical; edit both together.")


if __name__ == "__main__":
    unittest.main()
