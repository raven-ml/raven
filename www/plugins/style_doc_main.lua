-- plugins/style_doc_main.lua

-- Get the <main> element. The selector is configured in the widget.
local main_element = HTML.select_one(page, config["selector"])

if not main_element then
  Log.debug("style_doc_main: main element not found with selector: " .. config["selector"])
  Plugin.exit()
end

-- Determine if this is a "doc" page that needs prose styling.
-- You can make this logic more sophisticated.
local is_doc_page = false

-- Check if it's in a /guides/ or /api/ subdirectory or is a markdown file
if Regex.match(page_file, "/guides/") or Regex.match(page_file, "/api/") or Regex.match(page_file, "\\.md$") then
  is_doc_page = true
end

-- Alternative: Check for specific page patterns if needed
-- local doc_page_regex = config["doc_page_regex"] -- Get from widget config
-- if doc_page_regex and Regex.match(page_file, doc_page_regex) then
--   is_doc_page = true
-- end

-- Add classes if it's a doc page
if is_doc_page then
  Log.debug("style_doc_main: Applying prose styling to " .. page_file)
  HTML.add_class(main_element, "max-w-4xl") -- Tailwind class
  HTML.add_class(main_element, "mx-auto")   -- Tailwind class
  HTML.add_class(main_element, "px-6")      -- Tailwind class
  HTML.add_class(main_element, "py-8")      -- Tailwind class
  HTML.add_class(main_element, "prose")     -- Tailwind Typography class
  -- Add any other classes specific to documentation pages
  HTML.add_class(main_element, "lg:prose-xl") -- Example: larger prose on large screens
else
  Log.debug("style_doc_main: Not a doc page, skipping prose styling for " .. page_file)
end