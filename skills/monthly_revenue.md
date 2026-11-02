---
name: monthly_revenue
description: 計算某個月已付款訂單的總金額與筆數（範例，請依實際欄位修改）
tools: [mongo_aggregate]
---
1. 使用者沒有指定月份時，先問清楚是哪一年哪一個月，不要自己猜。
2. 用 `mongo_aggregate` 查 `orders` collection（有多個 MongoDB 連線時，`source` 用銷售系統那個）：
   - `$match`：`status` 等於 `"paid"`，`created` 落在該月第一天 00:00 到下個月第一天 00:00（台灣時間 UTC+8，
     換成 UTC 是前一天 16:00，例如 2026-03 是 `{"$gte": {"$date": "2026-02-28T16:00:00Z"}, "$lt": {"$date": "2026-03-31T16:00:00Z"}}`）
   - `$group`：`{"_id": null, "total": {"$sum": "$amount"}, "count": {"$sum": 1}}`
3. 回答時列出：月份、已付款筆數、總金額（加千分位，單位：元）。
4. 查不到資料時，直接說該月沒有已付款訂單，不要推測原因。
