val apply_command : State.t -> Command.t -> State.t * Effect.t list
val apply_event : State.t -> Event.t -> State.t * Effect.t list
