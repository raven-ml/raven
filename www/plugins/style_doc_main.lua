-- plugins/style_doc_main.lua
local page_file_str = page_file

Log.info("style_doc_main: Plugin CALLED for page: " .. page_file_str)
Log.info("style_doc_main: Widget selector is: " .. config["selector"])

local main_element = HTML.select_one(page, config["selector"])

if not main_element then
  Log.info("style_doc_main: [PAGE: " .. page_file_str .. "] Main element NOT FOUND with selector: " .. config["selector"])
  Plugin.exit()
end
Log.info("style_doc_main: [PAGE: " .. page_file_str .. "] Main element found.")

local is_doc_page_flag = nil -- Use nil as false, "t" (or any non-nil) as true
local matched_condition = "none"

if Regex.match(page_file_str, "/guides/") ~= nil then
  Log.info("style_doc_main: [PAGE: " .. page_file_str .. "] Matched /guides/")
  matched_condition = "/guides/"
  is_doc_page_flag = "t" -- Set to a non-nil value
elseif Regex.match(page_file_str, "/api/") ~= nil then
  Log.info("style_doc_main: [PAGE: " .. page_file_str .. "] Matched /api/")
  matched_condition = "/api/"
  is_doc_page_flag = "t"
elseif Regex.match(page_file_str, "\\.md$") ~= nil then
  Log.info("style_doc_main: [PAGE: " .. page_file_str .. "] Matched \\.md$")
  matched_condition = "\\.md$"
  is_doc_page_flag = "t"
end

Log.info("style_doc_main: [PAGE: " .. page_file_str .. "] Final Matched condition: " .. matched_condition .. "; is_doc_page_flag = " .. Value.repr(is_doc_page_flag))

-- In Lua 2.5, if condition treats nil and false as false, anything else as true.
if is_doc_page_flag then -- This will be true if is_doc_page_flag is "t"
  Log.info("style_doc_main: [PAGE: " .. page_file_str .. "] APPLYING prose styling.")
  HTML.add_class(main_element, "max-w-4xl")
  HTML.add_class(main_element, "mx-auto")
  HTML.add_class(main_element, "px-6")
  HTML.add_class(main_element, "py-8")
  HTML.add_class(main_element, "prose")
  HTML.add_class(main_element, "lg:prose-xl")
  Log.info("style_doc_main: [PAGE: " .. page_file_str .. "] Classes supposedly added.")
else
  Log.info("style_doc_main: [PAGE: " .. page_file_str .. "] SKIPPING prose styling (is_doc_page_flag was nil).")
end