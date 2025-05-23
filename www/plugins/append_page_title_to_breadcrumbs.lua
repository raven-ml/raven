-- plugins/append_page_title_to_breadcrumbs.lua

-- Check what's available
Log.debug("page_file: " .. tostring(page_file))
Log.debug("config selector: " .. tostring(config and config.selector))

if not page then
  Log.debug("page is nil")
  Plugin.exit()
end

if not config then
  Log.debug("config is nil") 
  Plugin.exit()
end

local breadcrumbs_container = HTML.select_one(page, config.selector)

if not breadcrumbs_container then
  Log.debug("Breadcrumbs container not found")
  Plugin.exit()
end

-- Get current page title from H1
local current_page_title = "Untitled"
local h1_element = HTML.select_one(page, "h1")
if h1_element then
  local h1_text = HTML.inner_text(h1_element)
  if h1_text then
    current_page_title = h1_text
  end
end

-- Add separator
local separator = HTML.create_element("span")
HTML.add_class(separator, "mx-2")
HTML.add_class(separator, "text-gray-400")
HTML.append_child(separator, HTML.create_text("/"))
HTML.append_child(breadcrumbs_container, separator)

-- Add the current page title
local title_span = HTML.create_element("span")
HTML.add_class(title_span, "text-gray-700")
HTML.add_class(title_span, "font-semibold")
HTML.append_child(title_span, HTML.create_text(current_page_title))
HTML.append_child(breadcrumbs_container, title_span)

Log.debug("Added breadcrumbs for: " .. current_page_title)