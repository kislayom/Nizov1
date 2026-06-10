package ai.nizo.tools.finance.model;

import java.time.LocalDate;

/**
 * One day's OHLCV bar for a ticker. All numeric fields are doubles to keep the JSON
 * serialization simple (BigDecimal noise isn't worth it for chart data the front-end
 * will floor to 2-4 decimals anyway).
 */
public record HistoricalPrice(
        LocalDate date,
        double open,
        double high,
        double low,
        double close,
        double adjustedClose,
        long volume
) implements java.io.Serializable {

    /** Daily change percent (close vs open). Returns 0 on degenerate input. */
    public double changePercent() {
        if (open == 0.0) return 0.0;
        return (close - open) / open * 100.0;
    }

    /** Bar range (high - low). */
    public double range() {
        return high - low;
    }
}
