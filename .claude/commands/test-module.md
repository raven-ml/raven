Create comprehensive tests for the module: $ARGUMENTS

Follow these steps:

1. Search for the module implementation file using Glob
2. Read and understand the module's public interface (.mli file if it exists)
3. Identify all exported functions, types, and values
4. Look for existing test patterns in the project's test directories
5. Create a test file following the project's naming convention (test_<module>.ml)
6. Write tests covering:
   - Basic functionality for each function
   - Edge cases (empty inputs, boundary values)
   - Error conditions and exceptions
   - Property-based tests for numeric operations if applicable
7. Run the tests with dune to ensure they pass
8. Format the code with `dune build @fmt --auto-promote`

Remember to follow the Alcotest framework patterns used in this project.