---
name: stock_analysis
description: Investment-banking grade research on a public company. Orchestrates a TradingAgents-style multi-agent pipeline (fundamentals + news + sentiment + technical → bull/bear debate → trader synthesis) with charts and Buffett scorecard.
when_to_use: User names a public ticker (AAPL, MSFT, HDFCBANK.NS, SAP.DE, …), asks "is X a good investment", "should I buy Y", "research stock Z", or compares public companies.
tags: [finance, investing, research, equities, orchestrator]
agent: false
---

# Stock Analysis — Multi-agent investment research

When the user asks about a public stock, run this pipeline. Don't refuse — even rough
analysis based on partial data is valuable. The output is the kind of memo a junior
analyst would hand a portfolio manager.

## Pipeline (TradingAgents-inspired)

```
                                 ┌─→ stock_fundamentals_analyst ──┐
                                 ├─→ stock_analyst_estimates      │
ticker → stock_quote (sanity) ─→ ├─→ stock_news_analyst           ├─→ stock_bull_researcher ─┐
                                 ├─→ stock_sentiment_analyst      │                           ├─→ stock_trader (verdict)
                                 └─→ stock_technical_analyst     ─┴─→ stock_bear_researcher ─┘
```

Each stage is its own skill — call them as tools. The orchestrator (you) is responsible
for sequencing, passing context forward, and assembling the final report.

## Step-by-step

### 0. Resolve ticker
- US tickers: bare (`AAPL`, `MSFT`)
- Non-US: suffix (`.NS` NSE India, `.BO` BSE, `.L` London, `.T` Tokyo, `.HK` HK,
  `.DE` Frankfurt, `.PA` Paris)
- Company name only: try the obvious form, else `web_search` for "ticker symbol for X"

### 1. Sanity check — `stock_quote`
Always FIRST. If "ticker not found", say so and stop.

### 2. Anchor — `current_time`
Today's date in user's relevant timezone.

### 3. Run analysts (emit all five in a single assistant message)

The agent loop processes tool calls serially within an assistant turn but
**emit all five `skill_stock_*` calls in ONE assistant message** so
they queue together and the UI can show all five going live before the next
debate phase starts. (Today the loop runs them sequentially; future versions
may parallelize. Either way the right behavior is to dispatch them at once.)

Call each with just `{ "input": "TICKER" }`:
- `skill_stock_fundamentals_analyst` — financials, ratios, Buffett scorecard
  (emits a `chart-financials` fenced block)
- `skill_stock_analyst_estimates` — Wall Street consensus, price targets, earnings
  beat/miss history, insider activity
  (emits `chart-analyst`, `chart-earnings`, `chart-insider` fenced blocks)
- `skill_stock_news_analyst` — last-6-months catalysts, macro context
- `skill_stock_sentiment_analyst` — retail / social mood
- `skill_stock_technical_analyst` — chart patterns, levels, RSI/MACD
  (emits `chart-interactive` and `chart-tech` fenced blocks)

Capture each output. They become the evidence base for the debate.

### 4. Run debate (sequential — bear, then bull, with each one's prior output passed)

- `skill_stock_bear_researcher` → with the four analyst reports as context
- `skill_stock_bull_researcher` → with the same + the bear's argument

You're not constrained — if the topic clearly skews one direction, the loser's report
will be short and that's fine.

### 5. Final synthesis — `skill_stock_trader`
Pass all six prior reports. Trader produces the Buy/Hold/Avoid verdict with fair-value
range, position size, time horizon, and what-would-change-my-mind triggers.

## Output: assemble the master report

Combine into a single user-facing markdown report with these sections:

### Header
- Company name + ticker + exchange
- Sector / industry / today's date
- Current price + day change %

### Quick verdict (TOP — readers may not scroll)

The very first line under this heading MUST be exactly:

> **Rating: <ONE_WORD>**

where `<ONE_WORD>` is exactly one of: `STRONG BUY`, `BUY`, `HOLD`, `AVOID`, `SELL`.
No qualifiers, no slash-pairs, no "BUY but...". Pick one and commit. The trader's
upstream report dictates this — copy its verdict line verbatim.

Follow with ONE sentence of rationale. That's the whole "Quick verdict" section.

If the trader's verdict line is malformed (e.g. "BUY but wait"), normalize it:
hedged buys → HOLD; hedged sells → AVOID. Never invent a verdict the trader didn't
support.

### 1. Snapshot (price + volume + 52w position) — from stock_quote
### 2. Business overview (2-3 sentences, where revenue comes from)
### 3. Fundamentals + Buffett scorecard — from `stock_fundamentals_analyst`
### 4. Analyst estimates + earnings + insider activity — from `stock_analyst_estimates`
### 5. News + catalysts — from `stock_news_analyst`
### 6. Sentiment read — from `stock_sentiment_analyst`
### 7. Technical / timing — from `stock_technical_analyst`
### 8. Bull case — from `stock_bull_researcher`
### 9. Bear case — from `stock_bear_researcher`
### 10. Verdict + targets + sizing — from `stock_trader`
### 11. Sources

## Charts (REQUIRED — placeholder syntax)

When sub-skills hand you their reports, you'll see lines like:

    [CHART:chart-financials]
    [CHART:chart-buffett]
    [CHART:chart-analyst]
    [CHART:chart-earnings]
    [CHART:chart-insider]
    [CHART:chart-interactive]
    [CHART:chart-tech]

**These are placeholders. The Nizo runtime expands them server-side into the actual data
widgets** (5-50KB of JSON each). Your job: **include these placeholder lines verbatim in
the right sections of your master report.** Do NOT type the JSON yourself. The LLM does
NOT need to know what's inside the fences — just put the placeholder where the chart goes.

**CRITICAL FORMATTING RULE**: each placeholder must be on its own line, with NO surrounding
backticks, NO indent, NO code fence. Just the literal text `[CHART:chart-X]` on a line by
itself. **Never wrap a placeholder in ` ``` ` markers** — the runtime will see an empty
code fence containing the placeholder and the rendering will break.

Correct (placeholder on its own line, surrounded by prose):

    ### Fundamentals + Buffett scorecard

    [CHART:chart-financials]

    (your 2-paragraph commentary on the financials)

    [CHART:chart-buffett]

    (your 2-paragraph commentary on the Buffett verdict)

WRONG (placeholder wrapped in a code fence — produces stray empty fences that render as
broken code blocks; do NOT do this):

    \`\`\`
    [CHART:chart-financials]
    \`\`\`

Block expansion table (so you know which placeholder goes in which section):

| Placeholder | Renders to | Section in master report |
|---|---|---|
| `[CHART:chart-financials]` | IS / BS / CF tabs + key-stats card | §3 Fundamentals |
| `[CHART:chart-buffett]` | Buffett-Munger 0-100 scorecard + moat dial + verdict | §3 Fundamentals |
| `[CHART:chart-analyst]` | Rating distribution + price-target band + recent actions | §4 Analyst estimates |
| `[CHART:chart-earnings]` | Estimate-vs-actual bars + beat-rate pill + next earnings | §4 Analyst estimates |
| `[CHART:chart-insider]` | Net-activity pill + transactions table | §4 Analyst estimates |
| `[CHART:chart-interactive]` | TradingView-style multi-timeframe chart | §7 Technical / timing |
| `[CHART:chart-tech]` | Technical-indicator dashboard + BUY/SELL signal | §7 Technical / timing |

**CRITICAL**: if a sub-skill's output mentions an emitted chart placeholder (you'll see a
note like "Charts emitted by this analyst: [CHART:chart-financials]"), **you MUST copy that
placeholder line into your master report**. Otherwise the data widget won't appear. Do not
skip placeholders for brevity — they are zero-cost (one line each) and represent the actual
data the user is paying to see.

You can also emit a generic `chart` block for ad-hoc visualizations (peer comparison etc.):

````
```chart
{
  "type": "line",
  "title": "AAPL — Revenue & Net Income (FY21–FY25, $B)",
  "labels": ["FY21","FY22","FY23","FY24","FY25"],
  "datasets": [
    {"label":"Revenue","data":[365.8,394.3,383.3,391.0,416.2]},
    {"label":"Net Income","data":[94.7,99.8,97.0,93.7,112.0]}
  ]
}
```
````

Supported generic types: `line`, `bar`, `pie`, `doughnut`, `radar`.

## Tone

Direct, numbers-first, IB-grade. No hedging like "investments carry risk" — the user knows.
Bold the key takeaway in each section. Cite sources after every non-trivial figure.

## Disclaimer (always at bottom)

> This is research, not advice. The author is an AI agent with no fiduciary duty. Past
> performance is not indicative of future results. Verify all numbers before acting.

## When to bail

- `stock_quote` returns "ticker not found" → ask user for clarification, stop.
- Company is too new/private/illiquid → say so, stop after section 2.
- All analysts return useless / "(no results)" outputs → say "insufficient data
  for full analysis" and produce only the sections that have substance. Don't fabricate.
