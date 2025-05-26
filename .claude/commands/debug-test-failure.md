Debug a failing test: $ARGUMENTS

Systematic debugging approach:

1. Run the specific test with verbose output
2. Read the test file to understand what it's testing
3. Examine the error message and stack trace carefully
4. Search for the implementation of the failing function
5. Check recent git commits that might have introduced the issue
6. Add debug print statements if needed (remember to remove them later)
7. Consider these common issues in OCaml/tensor code:
   - Type mismatches or phantom type issues
   - Incorrect tensor shapes or broadcasting
   - View/copy semantics confusion
   - Effect handler nesting problems (for Rune)
   - Backend-specific numerical precision differences
8. Write a minimal reproduction case if the issue is complex
9. Fix the issue and verify all related tests still pass
10. Clean up any debug code and format with `dune build @fmt --auto-promote`

Use `dune utop` for interactive debugging when needed.