# Function Calling Agent（內部 LLM API + MongoDB + API Trigger）

用 function calling 建立的 agent：語言模型走公司內部 API，工具包含「查詢 MongoDB」與「觸發內部 API」。
只依賴 `requests`、`pymongo`、`PyYAML`，不綁任何框架，方便部署到內網。

## 架構

```
使用者問題
   │
   ▼
Agent loop (agent/agent.py)
   │  messages + tools schema
   ▼
LLMClient (agent/llm_client.py) ──► 內部 LLM API  /chat/completions
   │  回傳 tool_calls
   ▼
ToolRegistry (agent/tools/registry.py) ── 驗證參數 / 需確認的工具先詢問使用者
   ├── MongoTools  (mongo_tools.py)  唯讀：list / schema / find / count / aggregate
   └── ApiEndpoint (api_tools.py)    設定檔中每個 API 各自成為一個工具
   │  工具結果（JSON 字串）
   ▼
回到 LLM，直到模型給出最終回答或達到 max_steps
```

## 快速開始

```bash
cd LLM
pip install -r requirements.txt
cp config.example.yaml config.yaml      # 修改 MongoDB 資料庫、可查的 collection、API 清單

export LLM_BASE_URL=http://llm.internal:8000/v1
export LLM_MODEL=your-model
export LLM_API_KEY=xxx                  # 不需要可不設
export MONGO_USER=readonly_user MONGO_PW=xxx MONGO_IP=10.0.0.5
export ORDER_API_TOKEN=... JOB_API_TOKEN=...

python main.py                          # 互動模式
python main.py -q "上個月已付款訂單總金額是多少？" -v
```

### 以 FastAPI 服務執行

```bash
export AGENT_API_KEY=自訂一組密鑰        # 建議設定；設定後每個請求都要帶 X-API-Key
uvicorn server:app --host 0.0.0.0 --port 8000
# 或 python server.py（可用 PORT、HOST 環境變數）
```

啟動後可打開 `http://localhost:8000/docs` 看互動式 API 文件並直接測試。

| 方法 | 路徑 | 用途 |
|---|---|---|
| GET | `/health` | 健康檢查（不需金鑰） |
| GET | `/tools` | 列出 agent 可用的工具 |
| POST | `/chat` | 問問題 |
| DELETE | `/sessions/{session_id}` | 清除某段對話記錄 |

```bash
curl -X POST http://localhost:8000/chat \
  -H "X-API-Key: $AGENT_API_KEY" -H "Content-Type: application/json" \
  -d '{"message": "上個月有幾筆已付款訂單？"}'
```

回應：

```json
{
  "session_id": "13fa3cdc...",
  "answer": "上個月已付款訂單共 2 筆。",
  "tool_calls": [{"name": "mongo_count", "arguments": {...}, "result": {"count": 2}}],
  "pending_actions": []
}
```

- **連續對話**：下一個請求帶上同一個 `session_id`，agent 會記得前面的內容。不帶就開新對話。對話存在記憶體，閒置超過 `SESSION_TTL_SECONDS`（預設 3600）會清除，服務重啟也會清空。
- **需要確認的 API**（非 GET 預設需要）：預設不執行，會出現在 `pending_actions`。確認後用同一個 `session_id` 再送一次，並帶上 `"allow_actions": ["trigger_report_job"]`（只允許這些工具）或 `"allow_actions": true`（允許這次請求的所有動作）。

| 環境變數 | 預設 | 說明 |
|---|---|---|
| `AGENT_CONFIG` | `config.yaml` | 設定檔路徑 |
| `AGENT_API_KEY` | 空（不檢查） | 請求需帶的 `X-API-Key` |
| `SESSION_TTL_SECONDS` | 3600 | 對話閒置多久後清除 |
| `MAX_SESSIONS` | 1000 | 記憶體中最多保留幾段對話 |

> 對話存在單一程序的記憶體中，請用單一 worker 執行（不要加 `--workers 2` 以上），否則同一段對話的請求可能送到不同程序而找不到記錄。

在程式中使用：

```python
from agent.config import build_agent, load_config

agent = build_agent(load_config("config.yaml"), confirm=lambda name, args: True)
print(agent.run("幫我查訂單 A123 的狀態，如果還沒出貨就觸發報表任務"))
```

## MongoDB 連線

### 用自己的連線套件（`Clinet_mongo(user, pw, ip, db)`）

1. 把套件放到 `main.py` 旁邊（例如 `mongo_client.py`），或用 `pip install` 安裝
2. 在 `config.yaml` 填：

```yaml
mongodb:
  factory: mongo_client:Clinet_mongo   # 「模組名:函式名」
  user: ${MONGO_USER}
  password: ${MONGO_PW}
  ip: ${MONGO_IP}
  database: mydb
  allowed_collections: [orders, customers]
```

程式會呼叫 `Clinet_mongo(user, password, ip, database)`，回傳值可以是：
- pymongo 的 `Database`
- pymongo 的 `MongoClient`（會自動取 `database`）
- 自訂類別，把上面兩種之一存在 `.db`、`.database`、`.client`、`.conn` 或 `.connection` 屬性

### 用連線字串

```yaml
mongodb:
  uri: ${MONGO_URI}
  database: mydb
```

## 內部 LLM API 的兩種模式（`llm.tool_mode`）

| 模式 | 適用情況 | 作法 |
|---|---|---|
| `native`（預設） | API 相容 OpenAI `/chat/completions` 且支援 `tools`（vLLM、TGI、Ollama、Azure、多數內部 gateway） | 直接送 `tools`，讀回 `tool_calls` |
| `prompt` | API 不支援 function calling | 把工具 schema 放進 system prompt，要求模型輸出 `{"tool_calls": [...]}` 或 `{"final": "..."}`，再由程式解析 |

如果內部 API 的格式和 OpenAI 不同（例如欄位名稱、包一層 `data`、用簽章驗證），繼承 `LLMClient` 並覆寫：
- `_post(payload)`：組 request、送出
- `_parse_native(data)`：把回應轉成 `LLMResponse(content, tool_calls)`

```python
class MyLLM(LLMClient):
    def _post(self, payload):
        r = self.session.post(self.url, json={"model": self.model, **payload},
                              headers={**self.headers, "X-Sign": sign(payload)})
        return r.json()["data"]
```

## 工具

### MongoDB（唯讀）

| 工具 | 用途 |
|---|---|
| `mongo_list_collections` | 列出可查詢的 collection |
| `mongo_sample_schema` | 欄位與型別 + 幾筆範例（讓模型先了解結構） |
| `mongo_find` | filter / projection / sort / limit / skip |
| `mongo_count` | 計數 |
| `mongo_aggregate` | 統計、分組、join |

日期、ObjectId 使用 Extended JSON：`{"created": {"$gte": {"$date": "2026-01-01T00:00:00Z"}}}`、`{"_id": {"$oid": "..."}}`。

### 觸發 API

在 `config.yaml` 的 `apis` 底下每加一個 endpoint，就多一個工具（名稱 = `name`，參數 = `parameters` JSON Schema）：

- `GET` / `DELETE`：參數放 query string；其他方法放 JSON body
- URL 中的 `{order_id}` 會用同名參數替換（自動 URL encode）
- `${VAR}` 從環境變數讀取，token 不必寫在設定檔
- 非 GET 預設 `require_confirmation: true`，執行前會呼叫 `confirm(name, args)`；CLI 會問 y/N

也可以直接用程式註冊自訂工具：

```python
from agent import Tool

agent.registry.register(Tool(
    name="send_alert", description="Send an alert to the on-call channel",
    parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
    func=lambda text: my_alert(text), require_confirmation=True,
))
```

## 安全設計

- **MongoDB 唯讀**：拒絕 `$out`、`$merge`、`$where`、`$function`、`$accumulator`；建議連線帳號也只給 read 權限
- **Collection 白名單**：`allowed_collections`
- **結果大小限制**：`max_limit`（筆數上限）、`max_time_ms`（查詢逾時）、`max_result_chars`（回傳給模型的字數上限）
- **API 白名單**：模型只能呼叫設定檔列出的 endpoint，無法打任意 URL
- **人工確認**：會改資料的 API 預設需確認
- **錯誤回饋**：工具錯誤以 `{"error": ...}` 回給模型，讓它修正參數重試，而不是讓程式中斷
- **步數上限**：`max_steps` 防止無窮迴圈

## 測試

```bash
pip install -r requirements.txt
python -m pytest -q tests
```

測試使用 `mongomock` 與假的 LLM / HTTP session，不需要真實的 MongoDB 或 LLM。
