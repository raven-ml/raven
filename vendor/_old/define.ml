open C.Types
open Ctypes

type 'a method_spec = {
  cmd : selector_t;
  typ : 'a fn;
  imp : object_t -> selector_t -> 'a;
  enc : string;
  runtime_lock : bool;
  thread_registration : bool;
}

type method_spec' = MethodSpec : 'a method_spec -> method_spec'
type 'a ivar_spec = { name : string; typ : 'a typ; enc : string }
type ivar_spec' = IvarSpec : 'a ivar_spec -> ivar_spec'

type 'a prop_spec = {
  name : string;
  typ : 'a Objc_type.t;
  retain : bool;
  copy : bool;
  readonly : bool;
  notify_change : bool;
}

type prop_spec' = PropSpec : 'a prop_spec -> prop_spec'

let method_spec ~cmd ~typ ~enc ?(runtime_lock = false)
    ?(thread_registration = false) imp =
  MethodSpec { cmd; typ; imp; enc; runtime_lock; thread_registration }

let ivar_spec ~name ~typ ~enc = IvarSpec { name; typ; enc }

let prop_spec ?(retain = true) ?(copy = false) ?(readonly = false)
    ?(notify_change = false) name ~typ =
  PropSpec { name; typ; retain; copy; readonly; notify_change }
