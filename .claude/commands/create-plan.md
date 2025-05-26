Create a detailed agentic plan for: $ARGUMENTS

Follow this structure:

1. First, read TODO.md to understand current priorities and context
2. Create a new plan file in .claude/plans/ with format:
   `YYYY-MM-DD-<task-description>.md`

The plan should include:

# Task: [Clear task title]

**IMPORTANT: This checklist must be kept updated throughout
implementation**

- [X] Step 1: [First implementation task]
- [X] Step 2: [Second implementation task]
- [ ] Step 3: [Testing task]
- [ ] Step 4: [Documentation/cleanup task]

---

## Objective
[What we're trying to achieve and why]

## Context
[Current state, related work, dependencies]

## Approach
[High-level strategy and design decisions]

## Implementation Steps
1. [Detailed step with specific files/functions to modify]
2. [Each step should be independently verifiable]
3. [Include test writing as explicit steps]

## Testing Strategy
- Unit tests: [What to test]
- Integration tests: [If applicable]
- Manual verification: [How to verify it works]

## Success Criteria
- [ ] All tests pass
- [ ] Code follows project conventions
- [ ] TODO.md updated with progress
- [ ] No regressions in existing functionality

## Risks & Mitigation
[Potential issues and how to handle them]

Remember to:
- Reference specific files and functions
- Consider edge cases and error handling
- Plan for incremental commits
- Update TODO.md after creating the plan