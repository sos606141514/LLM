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
export MONGO_URI="mongodb://readonly_user:pwd@mongo.internal:27017"
export ORDER_API_TOKEN=... JOB_API_TOKEN=...

python main.py                          # 互動模式
python main.py -q "上個月已付款訂單總金額是多少？" -v
```

在程式中使用：

```python
from agent.config import build_agent, load_config

agent = build_agent(load_config("config.yaml"), confirm=lambda name, args: True)
print(agent.run("幫我查訂單 A123 的狀態，如果還沒出貨就觸發報表任務"))
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
