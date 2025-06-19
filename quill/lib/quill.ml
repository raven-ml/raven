module Document = Document
module View = View
module Text = Text
module Execution = Execution
module Command = Command
module Effect = Effect
module Engine = Engine
module Diff = Diff
module Event = Event
module Markdown = Markdown
module Cursor = Cursor

let empty_document = Document.empty
let parse_markdown = Markdown.parse
let serialize_document = Markdown.serialize
let empty_editor = Engine.empty
let execute_command = Engine.execute
