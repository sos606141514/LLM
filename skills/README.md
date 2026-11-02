# Skills

每個 `.md` 檔是一個 skill（這個 README 不會被載入）。格式：

```markdown
---
name: skill_name              # 英數字、_、-
description: 一句話說明什麼時候用（模型靠這句判斷要不要載入）
tools: [mongo_aggregate]      # 選用：步驟會用到的工具，名稱打錯啟動時會報錯
---
步驟說明……
```

在 `config.yaml` 設定 `skills_dir: skills` 後生效。
