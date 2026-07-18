module Contract =
  Backend_contract.Make
    (Nx_backend)
    (struct
      let create_context = Nx_backend.create_context
    end)

let () = Windtrap.run "nx backend contract" (Contract.suite ())
