(** Environment metadata for discovery and configuration.

    Metadata describes environment properties like supported render modes,
    version information, and authorship. It follows Gymnasium's metadata
    convention, enabling environment registries and documentation generation.

    {1 Usage}

    Create metadata for a custom environment:
    {[
      let metadata =
        Metadata.default
        |> Metadata.with_version (Some "1.0.0")
        |> Metadata.with_description (Some "CartPole balancing task")
        |> Metadata.add_render_mode "human"
        |> Metadata.add_render_mode "rgb_array"
        |> Metadata.with_render_fps (Some 50)
    ]}

    Check render mode support:
    {[
      if Metadata.supports_render_mode "human" metadata then
        (* render in human mode *)
    ]} *)

type t = {
  render_modes : string list;
      (** Available rendering modes (e.g., "human", "rgb_array") *)
  render_fps : int option;
      (** Frames per second for rendering, if applicable *)
  authors : string list;  (** Environment authors or contributors *)
  description : string option;  (** Brief environment description *)
  version : string option;  (** Version string (e.g., "1.0.0") *)
  supported_vector_modes : string list;  (** Supported vectorization modes *)
  tags : string list;  (** Classification tags (e.g., "control", "atari") *)
  extra : Yojson.Safe.t option;  (** Additional custom metadata as JSON *)
}
(** Environment metadata record. *)

val default : t
(** [default] is the default metadata with all fields empty or None. *)

val add_render_mode : string -> t -> t
(** [add_render_mode mode metadata] adds [mode] to supported render modes.

    Common modes: "human" (display for humans), "rgb_array" (pixel array),
    "ansi" (text representation). *)

val supports_render_mode : string -> t -> bool
(** [supports_render_mode mode metadata] checks if [mode] is supported. *)

val with_render_fps : int option -> t -> t
(** [with_render_fps fps metadata] sets the rendering frame rate. *)

val with_description : string option -> t -> t
(** [with_description desc metadata] sets the environment description. *)

val with_version : string option -> t -> t
(** [with_version version metadata] sets the version string. *)

val add_author : string -> t -> t
(** [add_author name metadata] adds [name] to the authors list. *)

val add_tag : string -> t -> t
(** [add_tag tag metadata] adds [tag] to the tags list. *)

val set_tags : string list -> t -> t
(** [set_tags tags metadata] replaces all tags with [tags]. *)

val to_yojson : t -> Yojson.Safe.t
(** [to_yojson metadata] serializes [metadata] to JSON. *)

val of_yojson : Yojson.Safe.t -> (t, string) result
(** [of_yojson json] deserializes [metadata] from JSON.

    Returns [Error msg] if the JSON structure is invalid. *)
