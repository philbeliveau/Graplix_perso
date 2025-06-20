# Budget Enforcement Implementation

## Overview

This document describes the implementation of **pre-flight budget checking** that prevents API calls when budgets would be exceeded. The system has been enhanced to check budgets **BEFORE** making API calls, not after, ensuring that overspending is prevented rather than just warned about.

## Key Features Implemented

### 1. Pre-Flight Budget Checking
- **Real-time cost estimation** before API calls
- **Budget validation** against daily and monthly limits
- **API call prevention** when limits would be exceeded
- **Clear error messages** with budget information

### 2. Automatic Model Fallback
- **Intelligent model selection** when budget constraints are hit
- **Cost-based model ranking** for optimal alternatives
- **Capability-aware suggestions** (vision, JSON support, etc.)
- **Transparent fallback reporting** in responses

### 3. Emergency Stop Mechanism
- **Critical budget overrun detection** (configurable threshold)
- **Complete provider shutdown** when emergency limits exceeded
- **Automatic recovery** when usage returns to normal levels

### 4. Comprehensive Budget Monitoring
- **Real-time budget status** for all providers
- **Usage percentage tracking** with warning thresholds
- **Historical usage analysis** and trend reporting
- **Detailed cost breakdowns** by provider and model

## Implementation Details

### Configuration (`src/core/config.py`)

```python
class BudgetConfig(BaseModel):
    # Enforcement settings
    strict_budget_enforcement: bool = True
    auto_switch_to_cheaper_model: bool = False
    budget_warning_threshold: float = 0.8  # 80%
    
    # Daily limits per provider (USD)
    daily_budget_openai: float = 10.0
    daily_budget_anthropic: float = 10.0
    daily_budget_google: float = 10.0
    daily_budget_mistral: float = 10.0
    
    # Monthly limits per provider (USD)
    monthly_budget_openai: float = 100.0
    monthly_budget_anthropic: float = 100.0
    monthly_budget_google: float = 100.0
    monthly_budget_mistral: float = 100.0
    
    # Emergency settings
    enable_emergency_stop: bool = True
    emergency_stop_multiplier: float = 1.2  # 120% of limit
```

### Cost Tracker Enhancements (`src/llm/cost_tracker.py`)

#### New Methods Added:

**`estimate_cost_before_call()`**
```python
def estimate_cost_before_call(
    self,
    provider: str,
    model: str,
    estimated_input_tokens: Optional[int] = None,
    estimated_output_tokens: Optional[int] = None,
    safety_margin: float = 1.1
) -> float:
    """Estimate cost for an API call before making it"""
```

**`can_afford()`**
```python
def can_afford(
    self,
    provider: str,
    estimated_cost: float,
    budget_config: Optional[Any] = None,
    enforce_limits: bool = True
) -> BudgetCheckResult:
    """Check if we can afford an API call before making it"""
```

**`get_remaining_budget()`**
```python
def get_remaining_budget(
    self,
    provider: str,
    budget_config: Optional[Any] = None
) -> Dict[str, float]:
    """Get remaining budget for a provider"""
```

**`check_emergency_stop()`**
```python
def check_emergency_stop(
    self,
    provider: str,
    budget_config: Optional[Any] = None
) -> bool:
    """Check if emergency stop should be triggered"""
```

### LLM Service Integration (`src/llm/multimodal_llm_service.py`)

#### Pre-Flight Budget Checking:

```python
def _check_budget_before_call(
    self,
    model_key: str,
    estimated_input_tokens: Optional[int] = None,
    estimated_output_tokens: Optional[int] = None,
    enforce_strict_limits: Optional[bool] = None
) -> Dict[str, Any]:
    """Check budget constraints before making an API call"""
```

#### Integration in API Methods:

Both `extract_pii_from_text()` and `extract_pii_from_image()` now include:

1. **Pre-flight budget check** before making API calls
2. **Budget-based API call prevention** with clear error messages
3. **Warning injection** into successful responses
4. **Budget status reporting** in all responses

### API Integration Wrapper (`src/llm/api_integration_wrapper.py`)

#### Enhanced Features:

**Budget Status Reporting:**
```python
def get_budget_status(self, provider: Optional[str] = None) -> Dict[str, Any]:
    """Get comprehensive budget status for all or specific provider"""
```

**Automatic Model Suggestion:**
```python
def suggest_alternative_model(
    self, 
    preferred_model: str, 
    task_requirements: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """Suggest an alternative model when the preferred model exceeds budget"""
```

**Enhanced PII Extraction with Fallback:**
```python
def extract_pii_with_tracking(
    self,
    image_data: str,
    model_key: str,
    document_type: str = "document",
    user_id: Optional[str] = None,
    document_id: Optional[str] = None,
    allow_auto_fallback: bool = True,
    **kwargs
) -> Dict[str, Any]:
```

## Configuration Options

### Environment Variables

Budget limits can be configured via environment variables using the nested delimiter pattern:

```bash
# Enforcement settings
BUDGET__STRICT_BUDGET_ENFORCEMENT=true
BUDGET__AUTO_SWITCH_TO_CHEAPER_MODEL=false
BUDGET__BUDGET_WARNING_THRESHOLD=0.8

# Daily limits
BUDGET__DAILY_BUDGET_OPENAI=10.0
BUDGET__DAILY_BUDGET_ANTHROPIC=10.0
BUDGET__DAILY_BUDGET_GOOGLE=10.0
BUDGET__DAILY_BUDGET_MISTRAL=10.0

# Monthly limits
BUDGET__MONTHLY_BUDGET_OPENAI=100.0
BUDGET__MONTHLY_BUDGET_ANTHROPIC=100.0
BUDGET__MONTHLY_BUDGET_GOOGLE=100.0
BUDGET__MONTHLY_BUDGET_MISTRAL=100.0

# Emergency settings
BUDGET__ENABLE_EMERGENCY_STOP=true
BUDGET__EMERGENCY_STOP_MULTIPLIER=1.2
```

### Programmatic Configuration

```python
from core.config import BudgetConfig
from llm.api_integration_wrapper import MultiLLMIntegrationWrapper

# Create custom budget configuration
budget_config = BudgetConfig(
    strict_budget_enforcement=True,
    daily_budget_openai=5.0,  # $5 daily limit
    monthly_budget_openai=50.0  # $50 monthly limit
)

# Initialize wrapper with custom config
wrapper = MultiLLMIntegrationWrapper(budget_config=budget_config)
```

## Usage Examples

### Basic PII Extraction with Budget Enforcement

```python
from llm.api_integration_wrapper import llm_integration

# This call will be checked against budget limits before execution
result = llm_integration.extract_pii_with_tracking(
    image_data="base64_encoded_image",
    model_key="openai/gpt-4o",
    document_type="invoice"
)

if result["success"]:
    print("PII extracted successfully")
    if "budget_warnings" in result:
        print(f"Budget warnings: {result['budget_warnings']}")
else:
    print(f"Failed: {result['error']}")
    if "budget_status" in result:
        print(f"Budget status: {result['budget_status']}")
```

### Check Budget Status

```python
# Get comprehensive budget status
status = llm_integration.get_budget_status()

for provider, info in status["providers"].items():
    daily = info["daily"]
    print(f"{provider}: {daily['percentage_used']:.1f}% of daily budget used")
    
    if info["status"] == "warning":
        print(f"  ⚠️  Approaching budget limit")
    elif info["status"] == "budget_exceeded":
        print(f"  ❌ Budget exceeded")
    elif info["emergency_stop"]:
        print(f"  🚨 Emergency stop activated")
```

### Manual Cost Estimation

```python
from llm.cost_tracker import default_cost_tracker

# Estimate cost before making a call
estimated_cost = default_cost_tracker.estimate_cost_before_call(
    provider="openai",
    model="gpt-4o",
    estimated_input_tokens=1000,
    estimated_output_tokens=500
)

print(f"Estimated cost: ${estimated_cost:.6f}")

# Check if we can afford it
budget_check = default_cost_tracker.can_afford(
    provider="openai",
    estimated_cost=estimated_cost
)

if budget_check.can_afford:
    print("✅ API call allowed")
else:
    print(f"❌ API call blocked: {budget_check.blocking_reason}")
```

## Error Handling

### Budget Limit Exceeded

When a budget limit would be exceeded, the system returns:

```json
{
    "success": false,
    "error": "Budget limit exceeded: Daily budget exceeded: $0.0200 requested, $0.0100 remaining",
    "budget_check": {
        "can_afford": false,
        "estimated_cost": 0.02,
        "remaining_daily_budget": 0.01,
        "remaining_monthly_budget": 0.05,
        "daily_usage": 0.09,
        "monthly_usage": 0.15,
        "daily_limit": 0.10,
        "monthly_limit": 0.20,
        "warning_messages": [],
        "blocking_reason": "Daily budget exceeded: $0.0200 requested, $0.0100 remaining"
    }
}
```

### Emergency Stop Activated

When emergency stop is triggered:

```json
{
    "success": false,
    "error": "Budget limit exceeded: Emergency stop activated for openai due to critical budget overrun",
    "budget_check": {
        "can_afford": false,
        "blocking_reason": "Emergency stop activated for openai due to critical budget overrun"
    }
}
```

## Testing

### Automated Test Suite

Run the comprehensive test suite:

```bash
cd /path/to/pii_extraction_system
python testing/test_budget_enforcement.py
```

**Test Coverage:**
- ✅ Cost estimation accuracy
- ✅ Budget checking logic
- ✅ API call prevention
- ✅ Automatic model fallback
- ✅ Budget status reporting
- ✅ Emergency stop functionality

### Interactive Demonstration

Run the demonstration script:

```bash
python testing/demo_budget_enforcement.py
```

This shows real-world scenarios including:
- Normal API calls within budget
- Budget limit enforcement
- Automatic model suggestions
- Emergency stop triggers

## Benefits

### 1. **Prevents Overspending**
- No more surprise bills from runaway API usage
- Configurable daily and monthly limits per provider
- Real-time budget tracking and enforcement

### 2. **Intelligent Cost Management**
- Automatic fallback to cheaper models
- Cost-aware model recommendations
- Transparent cost reporting

### 3. **Operational Safety**
- Emergency stop mechanism for critical overruns
- Detailed budget monitoring and alerts
- Historical usage analysis for planning

### 4. **Developer Experience**
- Clear error messages with actionable information
- Comprehensive budget status in all responses
- Easy configuration via environment variables

## Migration Guide

### For Existing Code

No changes required for existing code. The budget enforcement is integrated transparently:

- All existing API calls will now include budget checking
- Budget warnings appear in successful responses
- Failed calls include budget status information

### For New Implementations

Use the enhanced API integration wrapper for full budget control:

```python
# Old approach (still works, but limited budget features)
from llm.multimodal_llm_service import llm_service
result = llm_service.extract_pii_from_image(...)

# New approach (full budget enforcement)
from llm.api_integration_wrapper import llm_integration
result = llm_integration.extract_pii_with_tracking(...)
```

## Monitoring and Alerts

### Budget Status Dashboard

Budget status can be integrated into monitoring dashboards:

```python
def get_budget_health_check():
    """Get budget health status for monitoring"""
    status = llm_integration.get_budget_status()
    
    alerts = []
    for provider, info in status["providers"].items():
        if info["emergency_stop"]:
            alerts.append(f"CRITICAL: Emergency stop for {provider}")
        elif info["status"] == "budget_exceeded":
            alerts.append(f"ERROR: Budget exceeded for {provider}")
        elif info["status"] == "warning":
            alerts.append(f"WARNING: {provider} approaching budget limit")
    
    return {
        "status": "healthy" if not alerts else "warning",
        "alerts": alerts,
        "details": status
    }
```

### Usage Analytics

Access detailed usage analytics:

```python
# Get cost analysis for the last 30 days
analysis = default_cost_tracker.get_cost_analysis(days=30)

print(f"Total cost: ${analysis['total_cost']:.2f}")
print(f"Average per request: ${analysis['avg_cost_per_request']:.4f}")

# Most expensive models
for model in analysis['expensive_models'][:5]:
    print(f"{model['model_name']}: ${model['total_cost']:.2f}")
```

## Security Considerations

### Budget Configuration Security

- Budget limits should be stored securely
- Consider using encrypted environment variables
- Implement role-based access for budget modifications

### Emergency Stop Protection

- Emergency stop thresholds should be carefully configured
- Monitor for potential budget manipulation attempts
- Implement audit logging for all budget-related decisions

## Conclusion

The budget enforcement implementation provides comprehensive cost control for the PII extraction system:

1. **Proactive cost control** through pre-flight budget checking
2. **Intelligent cost optimization** via automatic model fallback
3. **Operational safety** through emergency stop mechanisms
4. **Transparent reporting** with detailed budget status

The system ensures that API costs remain within configured limits while maintaining operational flexibility and user experience.