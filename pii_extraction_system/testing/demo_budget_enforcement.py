#!/usr/bin/env python3
"""
Budget Enforcement Demonstration

This script demonstrates the budget enforcement functionality in action,
showing how API calls are prevented when budgets would be exceeded.
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.config import BudgetConfig
from llm.cost_tracker import CostTracker
from llm.api_integration_wrapper import MultiLLMIntegrationWrapper

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demonstrate_budget_enforcement():
    """Demonstrate budget enforcement with realistic scenarios"""
    
    print("=" * 80)
    print("BUDGET ENFORCEMENT DEMONSTRATION")
    print("=" * 80)
    
    # Create test configuration with realistic but low limits
    demo_budget_config = BudgetConfig(
        strict_budget_enforcement=True,
        auto_switch_to_cheaper_model=True,
        budget_warning_threshold=0.8,  # 80% warning threshold
        daily_budget_openai=0.50,      # $0.50 daily limit
        daily_budget_anthropic=0.50,   # $0.50 daily limit
        daily_budget_google=0.50,      # $0.50 daily limit
        daily_budget_mistral=0.50,     # $0.50 daily limit
        monthly_budget_openai=10.0,    # $10.00 monthly limit
        monthly_budget_anthropic=10.0, # $10.00 monthly limit
        monthly_budget_google=10.0,    # $10.00 monthly limit
        monthly_budget_mistral=10.0,   # $10.00 monthly limit
    )
    
    # Initialize integration wrapper with budget enforcement
    session_id = f"demo_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wrapper = MultiLLMIntegrationWrapper(
        session_id=session_id,
        budget_config=demo_budget_config
    )
    
    print(f"\n📊 Budget Configuration:")
    print(f"   Daily limits: $0.50 per provider")
    print(f"   Monthly limits: $10.00 per provider")
    print(f"   Strict enforcement: {demo_budget_config.strict_budget_enforcement}")
    print(f"   Auto-switch to cheaper models: {demo_budget_config.auto_switch_to_cheaper_model}")
    
    # Show initial budget status
    print(f"\n💰 Initial Budget Status:")
    budget_status = wrapper.get_budget_status()
    for provider, status in budget_status.get('providers', {}).items():
        daily = status['daily']
        print(f"   {provider}: ${daily['usage']:.4f}/${daily['limit']:.2f} daily ({daily['percentage_used']:.1f}%)")
    
    print(f"\n🧪 Scenario 1: Normal API Call (Within Budget)")
    print("-" * 50)
    
    # Estimate cost for a normal API call
    if wrapper.cost_tracker:
        estimated_cost = wrapper.cost_tracker.estimate_cost_before_call(
            provider="openai",
            model="gpt-4o-mini",
            estimated_input_tokens=500,
            estimated_output_tokens=300
        )
        print(f"   Estimated cost for GPT-4o-mini: ${estimated_cost:.6f}")
        
        # Check if we can afford it
        budget_check = wrapper.cost_tracker.can_afford(
            provider="openai",
            estimated_cost=estimated_cost,
            budget_config=demo_budget_config
        )
        
        if budget_check.can_afford:
            print(f"   ✅ API call allowed: ${budget_check.remaining_daily_budget:.4f} daily budget remaining")
        else:
            print(f"   ❌ API call blocked: {budget_check.blocking_reason}")
    
    print(f"\n🧪 Scenario 2: Simulating Heavy Usage")
    print("-" * 50)
    
    # Simulate some usage to approach limits
    usage_records = [
        ("openai", "gpt-4o", 1000, 500, 0.025),
        ("openai", "gpt-4o", 1200, 600, 0.030),
        ("openai", "gpt-4o", 800, 400, 0.020),
        ("anthropic", "claude-3-5-sonnet-20241022", 1000, 500, 0.042),
    ]
    
    for provider, model, input_tokens, output_tokens, cost in usage_records:
        wrapper.cost_tracker.record_usage(
            provider=provider,
            model=model,
            task_type="demo_usage",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost=cost,
            success=True
        )
        print(f"   📝 Recorded: {provider}/{model} - ${cost:.4f}")
    
    # Show updated budget status
    print(f"\n💰 Updated Budget Status After Usage:")
    budget_status = wrapper.get_budget_status()
    for provider, status in budget_status.get('providers', {}).items():
        daily = status['daily']
        status_emoji = "🟢" if status['status'] == 'healthy' else "🟡" if status['status'] == 'warning' else "🔴"
        print(f"   {status_emoji} {provider}: ${daily['usage']:.4f}/${daily['limit']:.2f} daily ({daily['percentage_used']:.1f}%) - {status['status']}")
    
    print(f"\n🧪 Scenario 3: Attempting API Call That Would Exceed Budget")
    print("-" * 50)
    
    # Try to make an expensive call that would exceed budget
    expensive_cost = wrapper.cost_tracker.estimate_cost_before_call(
        provider="openai",
        model="gpt-4o",
        estimated_input_tokens=2000,
        estimated_output_tokens=1000
    )
    
    print(f"   Trying expensive call: ${expensive_cost:.6f}")
    
    budget_check = wrapper.cost_tracker.can_afford(
        provider="openai",
        estimated_cost=expensive_cost,
        budget_config=demo_budget_config
    )
    
    if budget_check.can_afford:
        print(f"   ✅ Expensive call allowed")
    else:
        print(f"   ❌ Expensive call blocked: {budget_check.blocking_reason}")
        
        # Try to suggest alternative
        alternative = wrapper.suggest_alternative_model(
            preferred_model="openai/gpt-4o",
            task_requirements={'requires_vision': True, 'requires_json': True}
        )
        
        if alternative:
            print(f"   💡 Suggested alternative: {alternative}")
            
            # Check if alternative would be affordable
            provider_alt, model_alt = alternative.split('/', 1)
            alt_cost = wrapper.cost_tracker.estimate_cost_before_call(
                provider=provider_alt,
                model=model_alt,
                estimated_input_tokens=2000,
                estimated_output_tokens=1000
            )
            
            alt_check = wrapper.cost_tracker.can_afford(
                provider=provider_alt,
                estimated_cost=alt_cost,
                budget_config=demo_budget_config
            )
            
            if alt_check.can_afford:
                print(f"   ✅ Alternative would be affordable: ${alt_cost:.6f}")
            else:
                print(f"   ❌ Alternative also exceeds budget: ${alt_cost:.6f}")
        else:
            print(f"   💔 No affordable alternative available")
    
    print(f"\n🧪 Scenario 4: Emergency Stop Simulation")
    print("-" * 50)
    
    # Simulate usage that triggers emergency stop
    wrapper.cost_tracker.record_usage(
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        task_type="emergency_demo",
        input_tokens=5000,
        output_tokens=2500,
        estimated_cost=0.70,  # This should trigger emergency stop
        success=True
    )
    
    emergency_stop = wrapper.cost_tracker.check_emergency_stop(
        provider="anthropic",
        budget_config=demo_budget_config
    )
    
    if emergency_stop:
        print(f"   🚨 EMERGENCY STOP triggered for Anthropic!")
        print(f"   🛑 All Anthropic API calls are now blocked")
    else:
        print(f"   ✅ No emergency stop triggered")
    
    # Final budget status
    print(f"\n💰 Final Budget Status:")
    budget_status = wrapper.get_budget_status()
    for provider, status in budget_status.get('providers', {}).items():
        daily = status['daily']
        monthly = status['monthly']
        status_emoji = "🟢" if status['status'] == 'healthy' else "🟡" if status['status'] == 'warning' else "🔴"
        emergency_icon = " 🚨" if status['emergency_stop'] else ""
        print(f"   {status_emoji} {provider}: Daily ${daily['usage']:.4f}/${daily['limit']:.2f} ({daily['percentage_used']:.1f}%), Monthly ${monthly['usage']:.4f}/${monthly['limit']:.2f} ({monthly['percentage_used']:.1f}%){emergency_icon}")
    
    print(f"\n📋 Key Takeaways:")
    print(f"   ✅ Budget enforcement prevents API calls when limits would be exceeded")
    print(f"   ✅ Real-time cost estimation helps predict budget impact")
    print(f"   ✅ Automatic fallback to cheaper models when configured")
    print(f"   ✅ Emergency stop prevents runaway costs")
    print(f"   ✅ Detailed budget status tracking and reporting")
    
    print(f"\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    
    # Cleanup test database
    try:
        if hasattr(wrapper.cost_tracker, 'db_path') and os.path.exists(wrapper.cost_tracker.db_path):
            os.remove(wrapper.cost_tracker.db_path)
            print("Test database cleaned up")
    except Exception as e:
        print(f"Warning: Could not clean up test database: {e}")


if __name__ == "__main__":
    demonstrate_budget_enforcement()