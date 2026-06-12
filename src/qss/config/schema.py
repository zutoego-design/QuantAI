from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ProjectConfig(BaseModel):
    name: str
    version: str
    timezone: str
    base_currency: str


class PathsConfig(BaseModel):
    raw_data: str
    silver_data: str
    gold_data: str
    reports: str


class RuntimeConfig(BaseModel):
    log_level: str = "INFO"
    overwrite_existing: bool = False
    cache_enabled: bool = True
    research_mode: bool = True
    allow_synthetic: bool = False


class PriceSourceConfig(BaseModel):
    provider: str
    adjusted_prices: bool = True
    fallback_provider: str | None = None
    batch_size: int = 25
    allow_synthetic_fallback: bool = False


class FundamentalsSourceConfig(BaseModel):
    provider: str
    use_xbrl: bool = True
    filing_types: list[str] = Field(default_factory=lambda: ["10-K", "10-Q"])
    user_agent: str = "QuantAI Research bot contact@example.com"
    fallback_to_synthetic: bool = False


class MacroSourceConfig(BaseModel):
    provider: str
    api_key_env_var: str | None = None


class ETFSourceConfig(BaseModel):
    provider: str
    tickers: dict[str, dict[str, str]]


class DataSourcesConfig(BaseModel):
    prices: PriceSourceConfig
    fundamentals: FundamentalsSourceConfig
    macro: MacroSourceConfig
    etf: ETFSourceConfig


class UniverseFiltersConfig(BaseModel):
    min_market_cap: float
    min_price: float
    min_adv_20d: float
    min_history_days: int
    min_price_data_completeness: float
    max_price_staleness_days: int = 7


class UniverseExcludeConfig(BaseModel):
    sectors: list[str] = Field(default_factory=list)
    security_types: list[str] = Field(default_factory=list)


class UniverseConfig(BaseModel):
    name: str
    market: str
    base_index_proxy: str
    seed_metadata_path: str
    filters: UniverseFiltersConfig
    exclude: UniverseExcludeConfig
    rebalance_frequency: str
    exchange: str = "XNAS"
    start_date: str = "2010-01-01"
    allowed_security_types: list[str] = Field(
        default_factory=lambda: ["Common Stock", "ADR", "REIT"]
    )
    long_history_provider: str = "alpha_vantage"
    validation_provider: str = "massive"
    recent_validation_years: int = 2
    min_recent_jaccard: float = 0.95
    min_recent_price_coverage: float = 0.98
    min_long_price_coverage: float = 0.95
    min_sector_coverage: float = 0.90
    max_remote_requests_per_sync: int = 25


class WinsorizeConfig(BaseModel):
    enabled: bool = True
    lower_quantile: float = 0.01
    upper_quantile: float = 0.99


class StandardizationConfig(BaseModel):
    method: str = "zscore"
    by_date: bool = True


class NeutralizationConfig(BaseModel):
    sector: bool = True
    market_cap: bool = True
    method: str = "cross_sectional_regression"


class FactorDefinition(BaseModel):
    weight: float
    direction: int

    @field_validator("direction")
    @classmethod
    def validate_direction(cls, value: int) -> int:
        if value not in (-1, 1):
            raise ValueError("direction must be -1 or 1")
        return value


class FactorGroupConfig(BaseModel):
    weight: float
    factors: dict[str, FactorDefinition]


class StrategyConfig(BaseModel):
    name: str
    rebalance_frequency: str
    risk_monitor_frequency: str
    benchmark: str
    min_factor_coverage: float = 0.80


class FactorProcessingConfig(BaseModel):
    winsorize: WinsorizeConfig
    standardization: StandardizationConfig
    neutralization: NeutralizationConfig


class OptimizerObjectiveConfig(BaseModel):
    maximize_alpha: bool = True
    risk_aversion: float = 5.0
    turnover_penalty: float = 1.0


class OptimizerConstraintConfig(BaseModel):
    long_only: bool = True
    fully_invested: bool = True
    min_weight: float = 0.0
    max_weight: float = 0.05
    target_num_holdings: int = 50
    max_sector_weight: float = 0.25
    max_turnover_per_rebalance: float = 0.30
    tracking_error_limit: float | None = 0.08


class CovarianceConfig(BaseModel):
    method: str = "historical_shrinkage"
    lookback_days: int = 252
    shrinkage_intensity: float = 0.20


class FallbackConfig(BaseModel):
    method: str = "top_n_equal_weight"
    top_n: int = 50


class OptimizerConfig(BaseModel):
    enabled: bool = True
    method: str
    candidate_count: int = 100
    objective: OptimizerObjectiveConfig
    constraints: OptimizerConstraintConfig
    covariance: CovarianceConfig
    fallback: FallbackConfig


class RiskPortfolioLimits(BaseModel):
    max_single_name_weight: float
    max_sector_weight: float
    max_daily_loss: float
    max_drawdown_alert: float
    max_realized_vol_annualized: float
    max_beta_to_benchmark: float
    min_beta_to_benchmark: float
    max_tracking_error: float
    max_turnover_monthly: float


class RiskAlertConfig(BaseModel):
    write_csv: bool = True
    write_html: bool = True
    send_email: bool = False


class RiskLimitsConfig(BaseModel):
    benchmark: str
    portfolio: RiskPortfolioLimits
    alerts: RiskAlertConfig


class MacroRuleInflationConfig(BaseModel):
    lookback_months: int = 6
    high_threshold_yoy: float = 0.03


class MacroRuleRatesConfig(BaseModel):
    curve_inversion_threshold: float = 0.0


class MacroRuleCreditConfig(BaseModel):
    spread_widening_zscore: float = 1.0


class MacroUsageConfig(BaseModel):
    affect_portfolio_weights: bool = False
    display_in_dashboard: bool = True
    include_in_risk_report: bool = True


class MacroConfig(BaseModel):
    fred_series: dict[str, str]
    etf_proxies: dict[str, str]
    regime_rules: dict[str, Any]
    usage: MacroUsageConfig


class TransactionCostConfig(BaseModel):
    commission_bps: float = 1.0
    slippage_bps: float = 5.0
    market_impact_coefficient: float = 0.10
    max_adv_participation: float = 0.10


class BacktestConfig(BaseModel):
    start_date: str
    end_date: str | None = None
    initial_capital: float = 1_000_000
    rebalance_frequency: str = "monthly"
    rebalance_execution_lag_days: int = 1
    transaction_cost: TransactionCostConfig
    delisting_return_scenarios: list[float] = Field(default_factory=lambda: [0.0, -0.30, -1.0])
    primary_benchmark: str = "^IXIC"
    secondary_benchmark: str = "QQQ"


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    includes: list[str] = Field(default_factory=list)
    project: ProjectConfig
    paths: PathsConfig
    runtime: RuntimeConfig
    data_sources: DataSourcesConfig
    universe: UniverseConfig
    strategy: StrategyConfig
    factor_processing: FactorProcessingConfig
    factor_groups: dict[str, FactorGroupConfig]
    optimizer: OptimizerConfig
    risk_limits: RiskLimitsConfig
    macro: MacroConfig
    backtest: BacktestConfig
