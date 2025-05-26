Implement a new tensor operation: $ARGUMENTS

Steps for adding a new operation to nx:

1. First, examine the backend interface at `nx/lib/core/backend_intf.ml`
2. Check if similar operations exist and follow their patterns
3. Add the operation in `nx/lib/core/frontend.ml`
4. Write sanity tests in `nx/test/test_nx_sanity.ml`
5. Run tests: `dune build @nx/runtest`
6. Format code: `dune fmt`

Follow the existing patterns for error handling and documentation.