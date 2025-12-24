-- RAG System Database Schema
-- For creating trade_history table

-- Note: Trends features are now retrieved directly from the features table
-- The features table has (stock_code, datetime) as primary key and contains all feature columns

-- Trade history table
CREATE TABLE IF NOT EXISTS trade_history (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  stock_code TEXT NOT NULL,
  trade_time TEXT NOT NULL,  -- ISO8601
  action TEXT NOT NULL,      -- 'BUY'/'SELL'/'HOLD' etc
  price REAL NOT NULL,
  volume REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_trade_stock_time ON trade_history(stock_code, trade_time);
CREATE INDEX IF NOT EXISTS idx_trade_stock ON trade_history(stock_code);

