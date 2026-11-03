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
| GET | `/skills` | 列出已載入的 skill |
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

### 直接連線（預設）

`config.yaml` 填帳號、密碼、IP、port、資料庫即可，不需要其他套件：

```yaml
mongodb:
  user: ${MONGO_USER}
  password: ${MONGO_PW}
  ip: ${MONGO_IP}
  port: 27017
  database: mydb
  auth_source: admin          # 帳號建立在哪個資料庫，通常是 admin；不確定就先用 admin
  allowed_collections: [orders, customers]
```

- 密碼裡有 `@`、`:` 等特殊字元也沒關係，不用另外編碼
- MongoDB 不需要登入時，把 `user`、`password` 刪掉
- 其他 pymongo 選項放在 `options`，例如 `options: {tls: true}`、`options: {replicaSet: rs0}`

### 用自己的連線套件（選用，例如 `mymongo.py` 裡的 class `Mongo`，用 `connectmongo()` 連線）

1. 把 `mymongo.py` 放到 `main.py` 旁邊（或用 `pip install` 安裝）
2. 在 `config.yaml` 填：

```yaml
mongodb:
  factory: mymongo:Mongo           # 「檔名:class 或函式名」
  connect_method: connectmongo     # 建立物件後要呼叫的連線方法；不需要就刪掉
  user: ${MONGO_USER}
  password: ${MONGO_PW}
  ip: ${MONGO_IP}
  port: 27017
  database: mydb
  allowed_collections: [orders, customers]
```

程式會執行：

```python
m = Mongo(user, password, ip, port, database)   # 沒寫 port 就是 Mongo(user, password, ip, database)
m.connectmongo()
```

帳密如果是傳給 `connectmongo` 而不是 `Mongo()`：

```yaml
  factory_args: []
  connect_args: [user, password, ip, port, database]
```

參數順序不同時，照你的順序排 `factory_args` / `connect_args`。

連線後，程式會在物件的屬性裡自動找出 pymongo 的 `MongoClient` 或 `Database`（屬性名稱不限），
或使用 `connectmongo()` 的回傳值。agent 只用這個連線做唯讀查詢，**不會呼叫你 class 裡的其他 function（例如儲存）**。

### 用連線字串

```yaml
mongodb:
  uri: ${MONGO_URI}
  database: mydb
```

### 連多個 MongoDB（用途不同）

把 `mongodb:` 改成清單，每個連線要有 `name`，並用 `description` 寫明用途：

```yaml
mongodb:
  - name: sales
    description: 銷售系統：訂單、客戶、付款
    user: ${SALES_MONGO_USER}
    password: ${SALES_MONGO_PW}
    ip: 10.0.0.5
    database: sales
    collections:                      # 每個 collection 一句說明，模型靠它選要查哪個
      orders: 訂單主檔，一筆一張訂單，含 order_no、金額、狀態、下單時間
      order_items: 訂單明細，一筆一個商品，用 order_no 對應 orders
      customers: 客戶基本資料
    notes: |                          # 給模型看的欄位意義與業務規則（選用）
      orders.amount 單位是新台幣元；營收只算 status = "paid"
      created_at 是 UTC

  - name: logistics
    description: 物流系統：出貨、倉庫、庫存
    uri: ${LOGISTICS_MONGO_URI}
    database: wms
    collections:
      shipments: 出貨紀錄，用 order_no 對應訂單
      stock: 各倉庫的即時庫存
    notes_file: notes/logistics.md    # 規則多時改放檔案（路徑相對於 config.yaml）
```

模型會先依連線的 `description` 選資料庫，再依 `collections` 的說明選 collection，例如問「訂單 A123 出貨了沒」
會去查 `logistics` 的 `shipments`。

- `collections` 同時是白名單：只有列出的 collection 能查，所以不要和 `allowed_collections` 一起寫
- 說明寫「一筆代表什麼」和「用哪個欄位對應其他 collection」，模型選得最準
- collection 總數不超過 `max_prompt_collections`（預設 30）時，說明直接列在 system prompt；超過時只列資料庫，
  模型用 `mongo_source_info` 讀取該資料庫的 collection 說明，避免 prompt 太長

- 每個連線的連線方式（帳密 / `uri` / `factory`）、`collections`（或 `allowed_collections`）、`max_limit`、`max_time_ms` 都各自設定
- 工具還是同一組（`mongo_find`、`mongo_count`…），只是多一個必填參數 `source`，值是連線的 `name`
- system prompt 只列出各連線的 `name` 和 `description`；`notes` 不會塞進 prompt，模型要查某個連線前才用 `mongo_source_info` 讀取，所以規則再多 prompt 也不會變長
- 只寫一個 `mongodb:`（不是清單）的舊設定照常可用，工具也不會多出 `source` 參數

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
| `mongo_source_info` | 某個連線的用途、各 collection 的說明、`notes`（有多個連線或有寫說明時才有） |
| `mongo_list_collections` | 列出可查詢的 collection（有寫說明時一併附上） |
| `mongo_sample_schema` | 欄位與型別 + 幾筆範例（讓模型先了解結構） |
| `mongo_find` | filter / projection / sort / limit / skip |
| `mongo_count` | 計數 |
| `mongo_aggregate` | 統計、分組、join |
| `load_skill` | 讀取某個 skill 的完整步驟（有設定 `skills_dir` 時才有） |

有多個連線時，每個工具都要帶 `source`。日期、ObjectId 使用 Extended JSON：`{"created": {"$gte": {"$date": "2026-01-01T00:00:00Z"}}}`、`{"_id": {"$oid": "..."}}`。

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

## Skills（給模型的作業步驟）

工具的 description 說明「這個工具做什麼」；skill 說明「某件事要怎麼做」，例如跨資料庫的報表算法、
呼叫 API 的固定流程。這些規則不屬於任何一個工具，放進 skill 管理。

1. 在 `skills/` 放 Markdown 檔，每個檔案一個 skill（範例：`skills/monthly_revenue.md`）：

   ```markdown
   ---
   name: refund_order
   description: 處理退款：查訂單、確認狀態、呼叫退款 API
   tools: [mongo_find, refund_api]   # 選用；寫了不存在的工具，啟動時就會報錯
   ---
   1. 用 mongo_find 查 sales 的 orders，確認 status 是 "paid"
   2. 已出貨的訂單不能直接退款，先告訴使用者
   3. 呼叫 refund_api ……
   ```

2. 在 `config.yaml` 加上 `skills_dir: skills`（路徑相對於 config.yaml）

運作方式：system prompt 只放每個 skill 的 `name` 和 `description`；模型判斷用得上時呼叫 `load_skill`
拿完整步驟再照做。skill 再多，prompt 也只多幾行。

| 規則類型 | 放哪裡 |
|---|---|
| 某個資料庫的欄位意義、單位 | 該連線的 `notes` / `notes_file` |
| 跨資料庫的算法、API 操作流程 | `skills/` |
| 每次都要遵守的總原則（語氣、語言） | `system_prompt` |

## 安全設計

- **MongoDB 唯讀**：拒絕 `$out`、`$merge`、`$where`、`$function`、`$accumulator`；建議連線帳號也只給 read 權限
- **Collection 白名單**：`collections` 或 `allowed_collections`
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
