-- post.lua: make POST /predict with JSON body

local function to_json(t)
  local parts = {}
  for k,v in pairs(t) do
    local val = (type(v) == "string") and ("\"" .. v .. "\"") or tostring(v)
    table.insert(parts, string.format("\"%s\":%s", k, val))
  end
  return "{" .. table.concat(parts, ",") .. "}"
end

wrk.method = "POST"
wrk.headers["Content-Type"] = "application/json"

-- Simple random helpers for variety
math.randomseed(os.time())
local function rnd(a, b) return math.floor((a + math.random() * (b - a)) * 100) / 100 end

request = function()
  local body_tbl = {
    ["alcohol"] = rnd(4.3, 7.9),
    ["malic_acid"] = rnd(1.0, 5.0),
    ["ash"] = rnd(1.5, 3.5),
    ["alcalinity_of_ash"] = rnd(10.0, 30.0),
    ["magnesium"] = rnd(70, 150),
    ["total_phenols"] = rnd(1.0, 4.0),
    ["flavanoids"] = rnd(0.5, 5.0),
    ["nonflavanoid_phenols"] = rnd(0.1, 0.7),
    ["proanthocyanins"] = rnd(0.5, 3.5),
    ["color_intensity"] = rnd(1.0, 13.0),
    ["hue"] = rnd(0.5, 2.0),
    ["od280/od315_of_diluted_wines"] = rnd(1.0, 4.0),
    ["proline"] = rnd(200, 1800)
  }
  local body = to_json(body_tbl)
  return wrk.format("POST", "/predict", nil, body)
end