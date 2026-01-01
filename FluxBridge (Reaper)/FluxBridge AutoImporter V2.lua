-- FluxBridge_AutoImporter_NEG.lua  (auto-mapper + AI importer)
-- Put this file in:  <REAPER resource path>/Scripts/FluxBridge/
-- Run it once from the Action List (stays alive until stopped).

local r = reaper
local SEP = package.config:sub(1, 1)

----------------------------------------------------------------------
-- paths & bootstrap
----------------------------------------------------------------------
local function path_join(a, b) if a:sub(-1) == SEP then return a .. b else return a .. SEP .. b end end
local function ensure_dir(p) r.RecursiveCreateDirectory(p, 0) end
local function file_exists(p) return r.file_exists(p) end
local function log(...) r.ShowConsoleMsg((table.concat({ ... }, "")) .. "\n") end

local INBOX = os.getenv("REAPER_BRIDGE_DIR")
if not INBOX or INBOX == "" then
  local appd = os.getenv("APPDATA")
  if appd and appd ~= "" then
    INBOX = path_join(appd, "REAPER" .. SEP .. "Scripts" .. SEP .. "FluxBridge" .. SEP ..
      "inbox")
  end
  if not INBOX or INBOX == "" or not file_exists(INBOX) then
    INBOX = path_join(r.GetResourcePath(), "Scripts" .. SEP .. "FluxBridge" .. SEP .. "inbox")
  end
end
local DONE_DIR = path_join(INBOX, "processed")
ensure_dir(INBOX); ensure_dir(DONE_DIR)

log("[FluxBridge NEG] listener running.")
log("Inbox: " .. INBOX)

do
  local f = io.open(path_join(INBOX, "BRIDGE_READY.txt"), "w")
  if f then
    f:write("inbox=", INBOX, "\n"); f:close()
  end
end

----------------------------------------------------------------------
-- tiny utils
----------------------------------------------------------------------
local function mv_to_processed(src, suffix)
  local base = src:match("[^" .. SEP .. "]+$") or ("job_" .. tostring(os.time()))
  local dst  = path_join(DONE_DIR, base .. "." .. (suffix or "done"))
  os.rename(src, dst)
  return dst
end

local function write_side_log(base_noext, lines)
  local f = io.open(base_noext .. ".log.txt", "w")
  if f then
    for _, ln in ipairs(lines) do f:write(ln, "\n") end
    f:close()
  end
end

local function safe_dofile(p)
  local ok, res = pcall(dofile, p)
  if not ok then return nil, ("parse fail: " .. tostring(res)) end
  return res
end

local function split_csv_line(ln)
  local out = {}
  for field in string.gmatch(ln, "([^,]+)") do out[#out + 1] = field end
  return out
end

local function read_csv_columns(path)
  local f, err = io.open(path, "r"); if not f then return nil, "open fail: " .. tostring(err) end
  local header = f:read("*l"); if not header then
    f:close(); return nil, "empty"
  end
  local names = split_csv_line(header)
  local cols = {}; for _, n in ipairs(names) do cols[n] = {} end
  for ln in f:lines() do
    local v = split_csv_line(ln)
    for i, n in ipairs(names) do
      local raw = v[i] or ""; local num = tonumber(raw); cols[n][#cols[n] + 1] = num or raw
    end
  end
  f:close(); return cols
end

-- mirror of the tracker's sanitize_name
local function sanitize(s)
  s = (s or ""):lower():gsub("[^%w_]+", "_"):gsub("_+", "_")
  return (s:match("^_*(.-)_*$") or s or "roi")
end

----------------------------------------------------------------------
-- track & FX helpers
----------------------------------------------------------------------
local function get_track_by_name(name)
  local n = r.CountTracks(0)
  for i = 0, n - 1 do
    local tr = r.GetTrack(0, i)
    local ok, nm = r.GetSetMediaTrackInfo_String(tr, "P_NAME", "", false)
    if ok and nm == name then return tr, i end
  end
  return nil, -1
end

local function ensure_track(name)
  local tr = select(1, get_track_by_name(name))
  if tr then return tr end
  local idx = r.CountTracks(0); r.InsertTrackAtIndex(idx, true)
  tr = r.GetTrack(0, idx); r.GetSetMediaTrackInfo_String(tr, "P_NAME", name, true)
  log("[track] created: ", name); return tr
end

-- Try MX first; fallback to legacy 5ch
local function ensure_jsfx_receiver(tr)
  local fx = r.TrackFX_GetByName(tr, "FluxBridge Receiver MX (18ch)", false)
  if fx < 0 then fx = r.TrackFX_GetByName(tr, "JS: FluxBridge_Receiver_MX", false) end
  if fx < 0 then fx = r.TrackFX_AddByName(tr, "JS: FluxBridge_Receiver_MX", false, -1) end
  if fx < 0 then
    fx = r.TrackFX_GetByName(tr, "FluxBridge Receiver (5ch)", false)
    if fx < 0 then fx = r.TrackFX_GetByName(tr, "JS: FluxBridge_Receiver", false) end
    if fx < 0 then fx = r.TrackFX_AddByName(tr, "JS: FluxBridge_Receiver", false, -1) end
  end
  if fx >= 0 then return fx else return nil, "JSFX receiver not found (place JS in Effects/)" end
end

-- [PATCH ROI_FOLDERS 1/3] ------------- folder + ROI child tracks ----------

local function set_folder_start(tr)
  if not tr then return end
  reaper.SetMediaTrackInfo_Value(tr, "I_FOLDERDEPTH", 1)
end

local function set_folder_end(tr)
  if not tr then return end
  reaper.SetMediaTrackInfo_Value(tr, "I_FOLDERDEPTH", -1)
end

local function set_folder_normal(tr)
  if not tr then return end
  reaper.SetMediaTrackInfo_Value(tr, "I_FOLDERDEPTH", 0)
end

local function find_top_level_insert_index()
  local r = reaper
  local n = r.CountTracks(0)
  if n == 0 then return 0 end
  local depth = 0
  local last_top = n - 1
  for i = 0, n - 1 do
    local tr = r.GetTrack(0, i)
    local d  = r.GetMediaTrackInfo_Value(tr, "I_FOLDERDEPTH") or 0
    depth    = depth + d
    if depth <= 0 then
      last_top = i
    end
  end
  return last_top + 1
end


-- Create (or reuse) the 3-track ROI bundle for a given base_name.
-- Cases:
--   1) No track named <base_name> exists:
--        <base_name>                 (folder shell)
--          <base_name> - SFX-Generator
--          <base_name> - Impacts IN
--          <base_name> - Impacts OUT  (closes folder)
--
--   2) Track named <base_name> exists AND has FX (old psycho/ROI track):
--        <existing parent ...>
--          <base_name>                 (NEW folder shell)
--            <base_name> - SFX-Generator  (this was the original track)
--            <base_name> - Impacts IN
--            <base_name> - Impacts OUT    (inherits old folder close)
--
--   3) Track named <base_name> exists but has NO FX (already a shell):
--        <base_name>                 (reused as folder shell)
--          <base_name> - SFX-Generator (created or reused)
--          <base_name> - Impacts IN
--          <base_name> - Impacts OUT  (does NOT close the higher folder)
function ensure_roi_tracks(base_name)
  local r = reaper
  base_name = tostring(base_name or "ROI")

  local parent = nil
  local sfx_tr, in_tr, out_tr = nil, nil, nil

  local base_tr, base_idx = get_track_by_name(base_name)

  local function create_child(at_idx, name)
    r.InsertTrackAtIndex(at_idx, true)
    local tr = r.GetTrack(0, at_idx)
    r.GetSetMediaTrackInfo_String(tr, "P_NAME", name, true)
    set_folder_normal(tr)
    return tr
  end

  local sfx_name = base_name .. " - SFX-Generator"
  local in_name  = base_name .. " - Impacts IN"
  local out_name = base_name .. " - Impacts OUT"

  if base_tr then
    -- existing track named <base_name>
    local fx_count = r.TrackFX_GetCount(base_tr)
    local old_depth = r.GetMediaTrackInfo_Value(base_tr, "I_FOLDERDEPTH") or 0

    if fx_count > 0 then
      -- wrap existing psycho/ROI track into a new folder
      r.GetSetMediaTrackInfo_String(base_tr, "P_NAME", sfx_name, true)

      r.InsertTrackAtIndex(base_idx, true)
      parent = r.GetTrack(0, base_idx)
      r.GetSetMediaTrackInfo_String(parent, "P_NAME", base_name, true)
      set_folder_start(parent)

      local sfx_idx = base_idx + 1
      sfx_tr = r.GetTrack(0, sfx_idx)
      set_folder_normal(sfx_tr)

      local insert_idx = sfx_idx + 1

      in_tr = select(1, get_track_by_name(in_name))
      if not in_tr then
        in_tr = create_child(insert_idx, in_name)
        insert_idx = insert_idx + 1
      end

      out_tr = select(1, get_track_by_name(out_name))
      if not out_tr then
        out_tr = create_child(insert_idx, out_name)
      end
      -- ALWAYS close the ROI folder here
      set_folder_end(out_tr)
    else
      -- base track is already a shell → reuse as folder
      parent            = base_tr
      base_tr, base_idx = get_track_by_name(base_name)
      set_folder_start(parent)

      local insert_idx = base_idx + 1

      sfx_tr = select(1, get_track_by_name(sfx_name))
      if not sfx_tr then
        sfx_tr = create_child(insert_idx, sfx_name)
        insert_idx = insert_idx + 1
      end

      in_tr = select(1, get_track_by_name(in_name))
      if not in_tr then
        in_tr = create_child(insert_idx, in_name)
        insert_idx = insert_idx + 1
      end

      out_tr = select(1, get_track_by_name(out_name))
      if not out_tr then
        out_tr = create_child(insert_idx, out_name)
      end
      -- LAST child: closes this ROI’s folder, not the outer parent
      set_folder_end(out_tr)
    end
  else
    -- brand new ROI: create its own top-level folder, not under the last parent
    local parent_idx = find_top_level_insert_index()
    r.InsertTrackAtIndex(parent_idx, true)
    parent = r.GetTrack(0, parent_idx)
    r.GetSetMediaTrackInfo_String(parent, "P_NAME", base_name, true)
    set_folder_start(parent)

    local insert_idx = parent_idx + 1

    sfx_tr = create_child(insert_idx, sfx_name)
    insert_idx = insert_idx + 1

    in_tr = create_child(insert_idx, in_name)
    insert_idx = insert_idx + 1

    out_tr = create_child(insert_idx, out_name)
    set_folder_end(out_tr)
  end

  -- receivers only on the SFX/Impact tracks, not the folder shell
  if ensure_roi_receiver_chain then
    ensure_roi_receiver_chain(sfx_tr)
    ensure_roi_receiver_chain(in_tr)
    ensure_roi_receiver_chain(out_tr)
  end
  ensure_jsfx_receiver(sfx_tr)
  ensure_jsfx_receiver(in_tr)
  ensure_jsfx_receiver(out_tr)

  return parent, sfx_tr, in_tr, out_tr
end

-- [PATCH IMPACT_CLEAR 1/2] --- clear envelope range for impact lanes --------
local function clear_env_range(env, t0, t1)
  if not env then return end
  -- delete all points (and automation-item points) in [t0, t1]
  reaper.DeleteEnvelopePointRangeEx(env, -1, t0, t1)
end

-- [PATCH IMPACT_MIDI 1/2] --- create MIDI items from a 0/1 impact lane ------

local IMPACT_PITCH_IN  = 36 -- C1
local IMPACT_PITCH_OUT = 36 -- same pitch; separate tracks distinguish role

local function clear_midi_range(track, t0, t1)
  if not track then return end
  local r = reaper
  local cnt = r.CountTrackMediaItems(track)
  for i = cnt - 1, 0, -1 do
    local it     = r.GetTrackMediaItem(track, i)
    local pos    = r.GetMediaItemInfo_Value(it, "D_POSITION")
    local len    = r.GetMediaItemInfo_Value(it, "D_LENGTH")
    local it_end = pos + len
    if it_end > t0 and pos < t1 then
      local tk = r.GetMediaItemTake(it, 0)
      if tk and r.TakeIsMIDI(tk) then
        r.DeleteTrackMediaItem(track, it)
      end
    end
  end
end

local function spawn_impact_midi(track, times, vals, t0, t1, is_out)
  local r = reaper
  if not track or not times or not vals or #times == 0 then return 0 end

  -- nuke old impact items in this time range
  clear_midi_range(track, t0, t1)

  local pitch = is_out and IMPACT_PITCH_OUT or IMPACT_PITCH_IN
  local num = 0
  local i = 1
  local n = #times
  local thr = 0.5

  while i <= n do
    local ti = times[i]
    local vi = tonumber(vals[i]) or 0
    if ti >= t0 and ti <= t1 and vi > thr then
      -- start of a segment
      local j = i + 1
      while j <= n do
        local tj = times[j]
        local vj = tonumber(vals[j]) or 0
        if tj > t1 or vj <= thr then break end
        j = j + 1
      end
      local t_start = ti
      local t_end   = times[j] or (ti + 0.020)
      if t_end <= t_start then
        t_end = t_start + 0.020
      end

      local item = r.CreateNewMIDIItemInProj(track, t_start, t_end, false)
      if item then
        local take = r.GetActiveTake(item)
        if take then
          local ppq0 = r.MIDI_GetPPQPosFromProjTime(take, t_start)
          local ppq1 = r.MIDI_GetPPQPosFromProjTime(take, t_end)
          r.MIDI_InsertNote(take, false, false, ppq0, ppq1, 0, pitch, 96, true)
          r.MIDI_Sort(take)
          num = num + 1
        end
      end
      i = j
    else
      i = i + 1
    end
  end
  if num > 0 then
    log(string.format("[midi] %d impact notes → %s", num,
      ({ reaper.GetSetMediaTrackInfo_String(track, "P_NAME", "", false) })[2] or "?"))
  end
  return num
end


-- [PATCH 1/3] --- load ROI Receiver FX chain on target track ------------------

-- helper: load an .RfxChain by file name from "<resource>/FX Chains/"
-- replace the helper completely
-- robust: load .RfxChain via API; fallback to chunk-injection; last resort: JS receiver
local function _replace_fxchain_in_chunk(track_chunk, fxchain_chunk)
  -- replace existing <FXCHAIN ...> ... '>' block (basic bracket counter, OK for empty tracks)
  local s = track_chunk:find("<FXCHAIN")
  if not s then
    -- insert right after the <TRACK ...> header line
    return track_chunk:gsub("(<TRACK[^\r\n]*\r?\n)", "%1" .. fxchain_chunk .. "\n", 1)
  end
  -- find closing '>' that balances nested '<...'
  local i = s
  local depth = 0
  local pos = s
  while true do
    local nl_start, nl_end, line = track_chunk:find("([^\r\n]*\r?\n)", pos)
    if not nl_start then break end
    local first = line:match("^%s*(.)") or ""
    if first == "<" then depth = depth + 1 end
    if line:match("^%s*>%s*\r?\n?$") then
      depth = depth - 1
      if depth == 0 then
        -- replace from 's' to the end of this line
        local head = track_chunk:sub(1, s - 1)
        local tail = track_chunk:sub(nl_end + 1)
        return head .. fxchain_chunk .. "\n" .. tail
      end
    end
    pos = nl_end + 1
  end
  return nil
end

local function load_fx_chain_by_name(tr, chain_name)
  if not tr or not chain_name or chain_name == "" then return false end
  local base = r.GetResourcePath()
  local candidates = {
    path_join(base, "FXChains" .. SEP .. chain_name .. ".RfxChain"),
    path_join(base, "FX Chains" .. SEP .. chain_name .. ".RfxChain"), -- rare, but seen in migrated installs
  }

  -- try official API load first
  local pre = r.TrackFX_GetCount(tr)
  for _, p in ipairs(candidates) do
    if file_exists(p) then
      local idx = r.TrackFX_AddByName(tr, p, false, -1)
      local post = r.TrackFX_GetCount(tr)
      if idx >= 0 and post > pre then return true end
    end
  end

  -- fallback: chunk injection (works even when AddByName fails)
  for _, p in ipairs(candidates) do
    if file_exists(p) then
      local f = io.open(p, "r"); if f then
        local raw = f:read("*a"); f:close()
        local fxchunk = raw:find("<FXCHAIN") and raw or ("<FXCHAIN\n" .. raw .. "\n>")
        local ok, tch = r.GetTrackStateChunk(tr, "", false)
        if ok and tch then
          local new = _replace_fxchain_in_chunk(tch, fxchunk)
          if new then
            local sOK = r.SetTrackStateChunk(tr, new, false)
            if sOK then return true end
          end
        end
      end
    end
  end

  -- last resort: at least ensure the receiver exists so envelopes have a target
  local idx = r.TrackFX_AddByName(tr, "JS: FluxBridge_Receiver_MX", false, -1)
  return idx >= 0
end

function ensure_roi_receiver_chain(tr)
  if r.TrackFX_GetCount(tr) == 0 then
    if load_fx_chain_by_name(tr, "ROI Receiver Template") then
      log("[fxchain] loaded: ROI Receiver Template")
      -- quick proof dump
      local c = r.TrackFX_GetCount(tr); log(("[fxchain] FX count now: %d"):format(c))
      for i = 0, c - 1 do
        local _, nm = r.TrackFX_GetFXName(tr, i, ""); log("  [fx] ", nm or "?")
      end
    else
      log("[fxchain] not found or inject failed; will fall back to JS receiver")
    end
  end
end

-- add tiny debug helper (optional)
local function _dump_track_fx(tr)
  local c = r.TrackFX_GetCount(tr)
  log(("[fxchain] FX count now: %d"):format(c))
  for i = 0, c - 1 do
    local ok, name = r.TrackFX_GetFXName(tr, i, "")
    log("  [fx] ", name or "?")
  end
end




-- Slider indices we expose. We accept multiple aliases per lane.
local SLIDER_INDEX = {
  env = 0,
  dirx = 1,
  diry = 2,
  posx = 3,
  posy = 4,
  rigid_abs = 5,
  rigid_env_abs = 5,
  rigid_rel = 6,
  rigid_env_rel = 6,
  dirz = 7,
  posz = 8,
  speed = 9,
  speed_xy = 9,
  speed_z = 10,
  acc = 11,
  acc_z = 12,
  jerk = 13,
  jerk_z = 14,
  impact_in = 15,
  impact_out = 16,
  impact_score = 17,
  pan = 18,
  pan_lr01 = 18,
  entropy = 19,
  axis_v   = 20,   -- |velocity along AoI|
  axis_acc = 21,   -- |accel along AoI|
  axis_jerk= 22,   -- |jerk along AoI|
  axis_dir = 23,   -- signed direction along AoI (0..1)
  lat_v = 24,
  lat_acc = 25,
  lat_jerk = 26,
  lat_dir = 27,
  lat_amp = 28,
}

local function ensure_env_for(tr, fxidx, key)
  local idx = SLIDER_INDEX[key]; if idx == nil then return nil end
  local env = r.GetFXEnvelope(tr, fxidx, idx, true)
  if env then r.Envelope_SortPoints(env) end
  return env
end

-- [PATCH 2/3] --- visibility policy for FX parameter envelopes ----------------

-- edit this table whenever you want new defaults:
local SHOW_BY_DEFAULT = {
  env = true,
  dirx = false,
  diry = false,
  dirz = false,
  impact_in = false,
  impact_out = false,
  impact_score = false,
  axis_v = false,
  axis_dir = true,
  -- everything not explicitly true here will be hidden by default
}

-- toggle envelope lane visibility by editing its chunk (no SWS dependency)
local function set_env_visible(env, show)
  if not env then return end
  local ok, chunk = r.GetEnvelopeStateChunk(env, "", false)
  if not ok or not chunk then return end
  -- flip first VIS flag only; keep rest of the line intact
  local new = chunk:gsub("VIS%s+%d+", "VIS " .. (show and "1" or "0"), 1)
  if new and new ~= chunk then r.SetEnvelopeStateChunk(env, new, false) end
end


local function insert_ai_series(env, t, v, t0, t1)
  if not env or not t or not v or #t == 0 then return 0 end
  local len = (t1 - t0); if len <= 0 then len = (t[#t] - t[1]) end
  if len <= 0 then len = 0.001 end
  local ai = r.InsertAutomationItem(env, -1, t0, len)
  local n = 0
  for i = 1, #t do
    local ti = t[i]; if type(ti) == "number" and ti >= t0 and ti <= t1 then
      local vi = tonumber(v[i]) or 0.0
      r.InsertEnvelopePointEx(env, ai, ti, vi, 0, 0, false, true)
      n = n + 1
    end
  end
  r.Envelope_SortPointsEx(env, ai); return n
end

----------------------------------------------------------------------
-- auto-mapping helpers (fill in missing columns by convention)
----------------------------------------------------------------------
local function map_aggregate_columns(cols)
  local m = {}
  if cols["flux_env"] then m.env = "flux_env" end
  if cols["rigid_env_abs"] then m.rigid_env_abs = "rigid_env_abs" end
  if cols["rigid_env_rel"] then m.rigid_env_rel = "rigid_env_rel" end
  if cols["agg_dirx01"] then m.dirx = "agg_dirx01" end
  if cols["agg_diry01"] then m.diry = "agg_diry01" end
  if cols["agg_posx01"] then m.posx = "agg_posx01" end
  if cols["agg_posy01"] then m.posy = "agg_posy01" end
  if cols["pan_lr01"] then m.pan = "pan_lr01" end
  -- optional pseudo-Z names if you add them later:
  if cols["agg_dirz01"] then m.dirz = "agg_dirz01" end
  if cols["agg_posz01"] then m.posz = "agg_posz01" end
  return m
end

local function map_roi_columns(cols, roi_name)
  local p = sanitize(roi_name or "roi")
  local m = {}
  if cols[p .. "_flux_env"] then m.env = p .. "_flux_env" end
  if cols[p .. "_rigid_env_abs"] then m.rigid_env_abs = p .. "_rigid_env_abs" end
  if cols[p .. "_rigid_env_rel"] then m.rigid_env_rel = p .. "_rigid_env_rel" end
  if cols[p .. "_dirx01"] then m.dirx = p .. "_dirx01" end
  if cols[p .. "_diry01"] then m.diry = p .. "_diry01" end
  if cols[p .. "_posx01"] then m.posx = p .. "_posx01" end
  if cols[p .. "_posy01"] then m.posy = p .. "_posy01" end
  if cols[p .. "_dirz01"] then m.dirz = p .. "_dirz01" end
  if cols[p .. "_posz01"] then m.posz = p .. "_posz01" end
  -- room to grow: speed/acc/jerk/impact etc if you export them later with the same prefix
  if cols[p .. "_speed01"] then m.speed = p .. "_speed01" end
  if cols[p .. "_acc01"] then m.acc = p .. "_acc01" end
  if cols[p .. "_jerk01"] then m.jerk = p .. "_jerk01" end
  if cols[p .. "_impact_out01"] then m.impact_out = p .. "_impact_out01" end
  if cols[p .. "_impact_in01"] then m.impact_in = p .. "_impact_in01" end
  if cols[p .. "_entropy01"] then m.entropy = p .. "_entropy01" end
   -- NEW: principal axis lanes (1-D along AoI)
  if cols[p .. "_axis_v11"]    then m.axis_v    = p .. "_axis_v11"    end
  if cols[p .. "_axis_acc11"]  then m.axis_acc  = p .. "_axis_acc11"  end
  if cols[p .. "_axis_jerk11"] then m.axis_jerk = p .. "_axis_jerk11" end
  if cols[p .. "_axis_dir11"]  then m.axis_dir  = p .. "_axis_dir11"  end
  if cols[p .. "_lat_v11"]    then m.lat_v    = p .. "_lat_v11"    end
  if cols[p .. "_lat_acc11"]  then m.lat_acc  = p .. "_lat_acc11"  end
  if cols[p .. "_lat_jerk11"] then m.lat_jerk = p .. "_lat_jerk11" end
  if cols[p .. "_lat_dir11"]  then m.lat_dir  = p .. "_lat_dir11"  end
  if cols[p .. "_lat_amp01"]  then m.lat_amp   = p .. "_lat_amp01"  end
  -- if cols[p.."_impact_active01"]        then m.impact_hold          = p.."_impact_in01" end
  -- OPTIONAL EXTENSION: auto-pick any columns that exactly match slider keys you add later
  -- e.g., if your CSV ever ships "roi_speed_z01" etc, just add branches above or rely on this pass-through:
  for k, _ in pairs(SLIDER_INDEX) do
    local candidate = p .. "_" .. k -- ex: roi_speed, roi_acc, etc
    if cols[candidate] and m[k] == nil then m[k] = candidate end
  end

  return m
end

----------------------------------------------------------------------
-- JOB processors
----------------------------------------------------------------------
local function process_catalog(req_path, job)
  local reply = job.reply
  local f = io.open(reply, "w")
  if not f then
    log("[catalog] cannot open reply: ", tostring(reply)); return
  end
  for i = 0, r.CountTracks(0) - 1 do
    local tr = r.GetTrack(0, i)
    local ok, nm = r.GetSetMediaTrackInfo_String(tr, "P_NAME", "", false)
    local alias = (nm or ("track_" .. i)):gsub("[^%w_]+", "_"):lower()
    f:write(string.format("track|%s|%s\n", alias, nm or ("Track " .. (i + 1))))
  end
  f:close(); log("[catalog] wrote: ", reply)
end

local function process_push(push_path, job)
  local csv = (job.csv or ""):gsub("\\", "/")
  local t0 = tonumber(job.start_sec or 0) or 0
  local t1 = tonumber(job.end_sec or t0) or t0

  log(string.format("[job] csv=%s  t0=%.3f  t1=%.3f  scene=%s", tostring(csv), t0, t1, tostring(job.scene_id)))

  local cols, err = read_csv_columns(csv)
  if not cols then
    log("[job] csv open fail: ", tostring(err)); return { ok = false, err = tostring(err) }
  end
  if not cols["time"] then
    log("[job] missing 'time' column"); return { ok = false, err = "no time column" }
  end
  local times = cols["time"]

  -- [PATCH ROI_FOLDERS 2/3] ------------- per-ROI bundle import -----------
  local function import_track(track_obj, is_agg)
    if not track_obj then return 0, 0 end

    local lanes, pts = 0, 0
    local tr_main, fx_main = nil, nil
    local tr_imp_in, tr_imp_out = nil, nil
    local fx_imp_in, fx_imp_out = nil, nil

    -- 1) Resolve track targets
    if is_agg then
      tr_main = ensure_track(tostring(track_obj.name or "Aggregate"))
      fx_main = select(1, ensure_jsfx_receiver(tr_main))
    else
      local parent, sfx_tr, in_tr, out_tr = ensure_roi_tracks(track_obj.name)
      tr_main                             = sfx_tr
      tr_imp_in                           = in_tr
      tr_imp_out                          = out_tr

      fx_main                             = select(1, ensure_jsfx_receiver(tr_main))
      fx_imp_in                           = select(1, ensure_jsfx_receiver(tr_imp_in))
      fx_imp_out                          = select(1, ensure_jsfx_receiver(tr_imp_out))
    end

    if not tr_main then return 0, 0 end
    if not fx_main then
      log("[fx] receiver missing on ", tostring(track_obj.name))
      return 0, 0
    end

    -- 2) Build mapping (explicit + auto)
    local mapping = {}
    for k, v in pairs(track_obj.columns or {}) do
      mapping[k] = v
    end

    local auto = is_agg and map_aggregate_columns(cols) or map_roi_columns(cols, track_obj.name)
    for k, v in pairs(auto) do
      if mapping[k] == nil then mapping[k] = v end
    end

    -- impact MIDI bookkeeping + param dedupe
    local impact_cols = { in_name = nil, out_name = nil }
    local used_param_for_idx = {} -- param index → first key that wrote it

    for key, colname in pairs(mapping) do
      local col = cols[colname]
      if not col then
        log(string.format("[env] skip %s.%s (no data)", tostring(track_obj.name), key))
        goto continue
      end

      -- choose target track / FX
      local target_tr = tr_main
      local target_fx = fx_main
      local is_impact = false

      if not is_agg then
        if key == "impact_in" then
          target_tr = tr_imp_in or tr_main
          target_fx = fx_imp_in or fx_main
          is_impact = true
          impact_cols.in_name = colname
        elseif key == "impact_out" or key == "impact_score" then
          target_tr = tr_imp_out or tr_main
          target_fx = fx_imp_out or fx_main
          is_impact = true
          if key == "impact_out" then
            impact_cols.out_name = colname
          end
        end
      end

      -- 3) De-duplicate by FX param index
      local param_idx = SLIDER_INDEX[key]
      if param_idx then
        local prev_key = used_param_for_idx[param_idx]
        if prev_key then
          log(string.format(
            "[env] skip %s.%s (FX param %d already written by '%s')",
            tostring(track_obj.name), key, param_idx, prev_key))
          goto continue
        end
        used_param_for_idx[param_idx] = key
      end

      -- 4) Ensure envelope
      local env = ensure_env_for(target_tr, target_fx, key)
      if not env then
        log(string.format("[env] skip %s.%s (no env)", tostring(track_obj.name), key))
        goto continue
      end

      -- 5) Hard overwrite [t0, t1] for this lane
      reaper.DeleteEnvelopePointRangeEx(env, -1, t0, t1)

      local n = insert_ai_series(env, times, col, t0, t1)
      pts     = pts + n
      lanes   = lanes + 1

      set_env_visible(env, SHOW_BY_DEFAULT[key] == true)

      local _, trname = reaper.GetSetMediaTrackInfo_String(target_tr, "P_NAME", "", false)
      log(string.format("[env] %s.%s +AI (%d pts) → %s",
        tostring(track_obj.name), key, n, trname or "?"))

      ::continue::
    end

    -- 6) Spawn MIDI for impact lanes (these helpers already clear old MIDI in-range)
    if not is_agg and impact_cols.in_name and tr_imp_in then
      spawn_impact_midi(tr_imp_in, times, cols[impact_cols.in_name], t0, t1, false)
    end
    if not is_agg and impact_cols.out_name and tr_imp_out then
      spawn_impact_midi(tr_imp_out, times, cols[impact_cols.out_name], t0, t1, true)
    end

    return lanes, pts
  end


  r.Undo_BeginBlock()

  local lanes, pts = 0, 0
  if job.agg_track then
    local l, p = import_track(job.agg_track, true); lanes = lanes + l; pts = pts + p
  end
  if job.roi_tracks then
    for _, t in ipairs(job.roi_tracks) do
      local l, p = import_track(t, false); lanes = lanes + l; pts = pts + p
    end
  end

  -- [PATCH 3/3] --- single baseline MIDI item spanning project content ----------

  local function project_content_range()
    local n = r.CountMediaItems(0)
    local earliest = math.huge
    local latest = 0.0
    for i = 0, n - 1 do
      local it = r.GetMediaItem(0, i)
      local pos = r.GetMediaItemInfo_Value(it, "D_POSITION")
      local len = r.GetMediaItemInfo_Value(it, "D_LENGTH")
      if pos < earliest then earliest = pos end
      if pos + len > latest then latest = pos + len end
    end
    if earliest == math.huge then earliest = 0.0 end
    if latest <= earliest then latest = r.GetProjectLength(0) end
    if latest <= earliest then latest = earliest + 1.0 end -- 1s fallback
    return earliest, latest
  end

  -- keep or reuse your existing project_content_range()

  -- constants
  local C1_PITCH = 36 -- assuming C4=60 naming; change if you use a different octave display

  local function track_has_any_midi(tr)
    if not tr then return false end
    local c = r.CountTrackMediaItems(tr)
    for i = 0, c - 1 do
      local it = r.GetTrackMediaItem(tr, i)
      local tk = r.GetMediaItemTake(it, 0)
      if tk and r.TakeIsMIDI(tk) then return true end
    end
    return false
  end

  local function ensure_midi_with_c1(tr, start_time, end_time)
    if not tr then return end
    if track_has_any_midi(tr) then
      log("[midi] track already has MIDI; skip baseline")
      return
    end
    local it = r.CreateNewMIDIItemInProj(tr, start_time, end_time, false) -- seconds
    if not it then return end
    local take = r.GetActiveTake(it); if not take then return end
    local ppq0 = r.MIDI_GetPPQPosFromProjTime(take, start_time)
    local ppq1 = r.MIDI_GetPPQPosFromProjTime(take, end_time)
    r.MIDI_InsertNote(take, false, false, ppq0, ppq1, 0, C1_PITCH, 96, true)
    r.MIDI_Sort(take)
    log(string.format("[midi] %s : C1 inserted %.3f..%.3f",
      ({ r.GetSetMediaTrackInfo_String(tr, "P_NAME", "", false) })[2] or "track", start_time, end_time))
  end

  local function project_content_range()
    local n = r.CountMediaItems(0)
    local earliest = math.huge
    local latest = 0.0
    for i = 0, n - 1 do
      local it = r.GetMediaItem(0, i)
      local pos = r.GetMediaItemInfo_Value(it, "D_POSITION")
      local len = r.GetMediaItemInfo_Value(it, "D_LENGTH")
      if pos < earliest then earliest = pos end
      if pos + len > latest then latest = pos + len end
    end
    if earliest == math.huge then earliest = 0.0 end
    if latest <= earliest then latest = r.GetProjectLength(0) end
    if latest <= earliest then latest = earliest + 1.0 end -- 1s fallback
    return earliest, latest
  end

  -- NEW: per-track spawner
  local function spawn_midi_for_all_tracks(job)
    local t0, t1 = project_content_range()

    -- Aggregate baseline
    if job.agg_track then
      local agg_name = tostring(job.agg_track.name)
      local tr = select(1, get_track_by_name(agg_name)) or ensure_track(agg_name)
      ensure_midi_with_c1(tr, t0, t1)
    end

    -- Per-ROI baseline only on SFX-Generator child
    if job.roi_tracks then
      for _, t in ipairs(job.roi_tracks) do
        local base = tostring(t.name)
        local sfx_name = base .. " - SFX-Generator"
        local tr = select(1, get_track_by_name(sfx_name))
        if not tr then
          -- fallback for older projects: use flat ROI track
          tr = select(1, get_track_by_name(base)) or ensure_track(base)
        end
        ensure_midi_with_c1(tr, t0, t1)
      end
    end
  end



  spawn_midi_for_all_tracks(job)
  r.Undo_EndBlock("[FluxBridge] Import motion", -1)
  return { ok = true, lanes = lanes, points = pts }
end

----------------------------------------------------------------------
-- main scanner
----------------------------------------------------------------------
local idx = 0
local function dispatch_one(full)
  local is_rq    = full:match("%.rq%.lua$")
  local is_rpush = full:match("%.rpush%.lua$")
  if not (is_rq or is_rpush) then return end
  log("[scan] found: ", full)

  local job, perr = safe_dofile(full)
  if not job then
    log("[parse] ", perr); mv_to_processed(full, "ERR_parse"); return
  end

  if is_rq then
    process_catalog(full, job); mv_to_processed(full, "done"); return
  end

  local result     = process_push(full, job)
  local moved      = mv_to_processed(full, result.ok and "done" or "ERR")

  local base_noext = moved:gsub("%.%w+$", "")
  local lines      = {
    os.date("[%Y-%m-%d %H:%M:%S]"),
    "csv=" .. tostring(job.csv),
    "scene_id=" .. tostring(job.scene_id),
    "t0=" .. tostring(job.start_sec) .. "  t1=" .. tostring(job.end_sec),
    "ok=" .. tostring(result.ok),
    "lanes=" .. tostring(result.lanes or 0),
    "points=" .. tostring(result.points or 0),
    result.err and ("err=" .. tostring(result.err)) or ""
  }
  write_side_log(base_noext, lines)
  if result.ok then
    log(string.format("[done] lanes=%d  points=%d  (%s)", result.lanes or 0, result.points or 0, moved))
  else
    log("[done] ERR. See log: ", moved .. ".log.txt")
  end
end

local function scan_once()
  while true do
    local fn = r.EnumerateFiles(INBOX, idx)
    if not fn then break end
    idx = idx + 1
    if fn:find("%.rq%.lua$") or fn:find("%.rpush%.lua$") then
      dispatch_one(path_join(INBOX, fn))
    end
  end
  idx = 0
end

local function loop()
  scan_once(); r.defer(loop)
end
loop()
