#!/usr/bin/env python3
"""
SPARC Orchestrator - Agent Coordination Monitor
Real-time monitoring and coordination of multi-agent PII extraction system development.
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import subprocess
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.layout import Layout
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

console = Console() if RICH_AVAILABLE else None


@dataclass
class AgentStatus:
    """Agent status tracking."""
    agent_id: str
    name: str
    role: str
    status: str  # "active", "idle", "blocked", "completed", "error"
    current_task: str
    progress_percentage: float
    last_update: datetime
    deliverables_completed: List[str]
    deliverables_pending: List[str]
    blocking_issues: List[str]
    dependencies_met: bool
    files_modified: List[str]
    test_status: str
    integration_status: str
    

class AgentCoordinationMonitor:
    """Monitor and coordinate multiple agents working on PII extraction system."""
    
    def __init__(self):
        """Initialize the coordination monitor."""
        self.project_root = Path(__file__).parent
        self.agents: Dict[str, AgentStatus] = {}
        self.start_time = datetime.now()
        self.monitoring_interval = 5  # seconds
        
        # Define agent specifications
        self.agent_specs = {
            "agent_1": {
                "name": "Environment & Infrastructure Specialist",
                "role": "CODER",
                "deliverables": [
                    "Project structure", "Configuration management", "Logging system",
                    "Document processing", "Data storage", "Core pipeline", "Rule-based extractor"
                ]
            },
            "agent_2": {
                "name": "PII Extraction Core Developer", 
                "role": "CODER + ANALYST",
                "deliverables": [
                    "NER extractor", "Dictionary extractor", "Evaluation metrics", "Integration tests"
                ]
            },
            "agent_3": {
                "name": "Advanced AI/ML Specialist",
                "role": "CODER + ANALYST", 
                "deliverables": [
                    "Layout-aware models", "Custom fine-tuning", "Ensemble methods", "Privacy integration"
                ]
            },
            "agent_4": {
                "name": "Frontend/Dashboard Developer",
                "role": "CODER",
                "deliverables": [
                    "Main dashboard app", "Document processing interface", "Model comparison", 
                    "Performance monitoring", "Data management tools"
                ]
            },
            "agent_5": {
                "name": "DevOps & CI/CD Specialist",
                "role": "CODER + DEVOPS",
                "deliverables": [
                    "Docker containerization", "CI/CD pipeline", "Infrastructure as Code", "Monitoring setup"
                ]
            },
            "agent_6": {
                "name": "Quality Assurance & Testing Lead",
                "role": "ANALYST + CODER",
                "deliverables": [
                    "Test data preparation", "Comprehensive test suite", "Quality metrics", "Security testing"
                ]
            },
            "agent_7": {
                "name": "Documentation & Integration Coordinator",
                "role": "GENERAL",
                "deliverables": [
                    "API documentation", "User documentation", "Integration testing", "Deployment docs"
                ]
            }
        }
        
        # Initialize agent statuses
        self._initialize_agent_statuses()
    
    def _initialize_agent_statuses(self):
        """Initialize agent status tracking."""
        for agent_id, spec in self.agent_specs.items():
            self.agents[agent_id] = AgentStatus(
                agent_id=agent_id,
                name=spec["name"],
                role=spec["role"],
                status="unknown",
                current_task="Initializing",
                progress_percentage=0.0,
                last_update=datetime.now(),
                deliverables_completed=[],
                deliverables_pending=spec["deliverables"],
                blocking_issues=[],
                dependencies_met=False,
                files_modified=[],
                test_status="pending",
                integration_status="pending"
            )
    
    def scan_file_system_changes(self) -> Dict[str, List[str]]:
        """Scan for recent file changes to detect agent activity."""
        changes_by_agent = {agent_id: [] for agent_id in self.agents.keys()}
        
        # Scan for recently modified files
        try:
            cutoff_time = datetime.now() - timedelta(minutes=30)
            
            for file_path in self.project_root.rglob("*.py"):
                if file_path.stat().st_mtime > cutoff_time.timestamp():
                    # Determine which agent likely modified this file
                    agent_id = self._infer_agent_from_file(file_path)
                    if agent_id:
                        changes_by_agent[agent_id].append(str(file_path.relative_to(self.project_root)))
                        
        except Exception as e:
            if console:
                console.print(f"[yellow]Warning: Error scanning files: {e}[/yellow]")
        
        return changes_by_agent
    
    def _infer_agent_from_file(self, file_path: Path) -> Optional[str]:
        """Infer which agent likely modified a file based on its location and content."""
        path_str = str(file_path)
        
        # Agent 2: PII Extraction Core
        if any(x in path_str for x in ["ner_extractor", "evaluation", "dictionary"]):
            return "agent_2"
        
        # Agent 3: Advanced AI/ML  
        if any(x in path_str for x in ["layout_aware", "training", "models", "ensemble"]):
            return "agent_3"
        
        # Agent 4: Dashboard
        if any(x in path_str for x in ["dashboard", "streamlit", "pages", "ui_"]):
            return "agent_4"
        
        # Agent 5: DevOps
        if any(x in path_str for x in ["Dockerfile", "docker-compose", "deploy", "monitoring"]):
            return "agent_5"
        
        # Agent 6: Testing
        if any(x in path_str for x in ["test_", "conftest", "fixtures", "performance"]):
            return "agent_6"
        
        # Agent 7: Documentation
        if any(x in path_str.lower() for x in ["doc", "readme", "guide", "api"]):
            return "agent_7"
        
        return None
    
    def check_agent_progress(self) -> None:
        """Check progress of all agents."""
        file_changes = self.scan_file_system_changes()
        
        for agent_id, agent in self.agents.items():
            # Update file modifications
            agent.files_modified = file_changes.get(agent_id, [])
            
            # Infer status based on activity
            if agent.files_modified:
                agent.status = "active"
                agent.last_update = datetime.now()
            elif agent.status == "active" and (datetime.now() - agent.last_update).seconds > 600:
                agent.status = "idle"
            
            # Check deliverables completion
            self._check_deliverables_completion(agent)
            
            # Check dependencies
            self._check_dependencies(agent)
            
            # Update progress percentage
            total_deliverables = len(agent.deliverables_completed) + len(agent.deliverables_pending)
            if total_deliverables > 0:
                agent.progress_percentage = (len(agent.deliverables_completed) / total_deliverables) * 100
    
    def _check_deliverables_completion(self, agent: AgentStatus) -> None:
        """Check which deliverables have been completed based on file system."""
        if agent.agent_id == "agent_1":
            # Check Agent 1 deliverables
            if (self.project_root / "pyproject.toml").exists():
                self._mark_deliverable_complete(agent, "Project structure")
            if (self.project_root / "src" / "core" / "config.py").exists():
                self._mark_deliverable_complete(agent, "Configuration management")
            if (self.project_root / "src" / "core" / "logging_config.py").exists():
                self._mark_deliverable_complete(agent, "Logging system")
            if (self.project_root / "src" / "utils" / "document_processor.py").exists():
                self._mark_deliverable_complete(agent, "Document processing")
            if (self.project_root / "src" / "utils" / "data_storage.py").exists():
                self._mark_deliverable_complete(agent, "Data storage")
            if (self.project_root / "src" / "core" / "pipeline.py").exists():
                self._mark_deliverable_complete(agent, "Core pipeline")
            if (self.project_root / "src" / "extractors" / "rule_based.py").exists():
                self._mark_deliverable_complete(agent, "Rule-based extractor")
        
        elif agent.agent_id == "agent_2":
            if (self.project_root / "src" / "extractors" / "ner_extractor.py").exists():
                self._mark_deliverable_complete(agent, "NER extractor")
            if (self.project_root / "src" / "extractors" / "evaluation.py").exists():
                self._mark_deliverable_complete(agent, "Evaluation metrics")
        
        elif agent.agent_id == "agent_3":
            if (self.project_root / "src" / "extractors" / "layout_aware.py").exists():
                self._mark_deliverable_complete(agent, "Layout-aware models")
            if (self.project_root / "src" / "models" / "training_pipeline.py").exists():
                self._mark_deliverable_complete(agent, "Custom fine-tuning")
        
        elif agent.agent_id == "agent_4":
            if (self.project_root / "src" / "dashboard" / "main.py").exists():
                self._mark_deliverable_complete(agent, "Main dashboard app")
        
        elif agent.agent_id == "agent_5":
            if (self.project_root / "Dockerfile").exists():
                self._mark_deliverable_complete(agent, "Docker containerization")
            if (self.project_root / "monitoring").exists():
                self._mark_deliverable_complete(agent, "Monitoring setup")
        
        elif agent.agent_id == "agent_6":
            if (self.project_root / "tests" / "conftest.py").exists():
                self._mark_deliverable_complete(agent, "Test data preparation")
            if list(self.project_root.glob("tests/**/test_*.py")):
                self._mark_deliverable_complete(agent, "Comprehensive test suite")
    
    def _mark_deliverable_complete(self, agent: AgentStatus, deliverable: str) -> None:
        """Mark a deliverable as complete."""
        if deliverable in agent.deliverables_pending:
            agent.deliverables_pending.remove(deliverable)
            agent.deliverables_completed.append(deliverable)
    
    def _check_dependencies(self, agent: AgentStatus) -> None:
        """Check if agent dependencies are met."""
        if agent.agent_id == "agent_2":
            # Depends on Agent 1 core infrastructure
            agent_1 = self.agents["agent_1"]
            agent.dependencies_met = len(agent_1.deliverables_completed) >= 5
        
        elif agent.agent_id == "agent_3":
            # Depends on Agent 2 baseline extractors
            agent_2 = self.agents["agent_2"]
            agent.dependencies_met = "NER extractor" in agent_2.deliverables_completed
        
        else:
            # Other agents can work in parallel
            agent.dependencies_met = True
    
    def generate_status_report(self) -> Dict[str, Any]:
        """Generate comprehensive status report."""
        total_agents = len(self.agents)
        active_agents = sum(1 for a in self.agents.values() if a.status == "active")
        completed_agents = sum(1 for a in self.agents.values() if a.progress_percentage >= 100)
        
        # Calculate overall progress
        total_progress = sum(a.progress_percentage for a in self.agents.values()) / total_agents
        
        # Identify blocking issues
        blocked_agents = [a for a in self.agents.values() if not a.dependencies_met or a.blocking_issues]
        
        # Integration readiness
        critical_components = ["Core pipeline", "Rule-based extractor", "NER extractor"]
        integration_ready = all(
            any(comp in a.deliverables_completed for a in self.agents.values())
            for comp in critical_components
        )
        
        return {
            "timestamp": datetime.now().isoformat(),
            "runtime": str(datetime.now() - self.start_time),
            "overall_progress": total_progress,
            "agent_summary": {
                "total": total_agents,
                "active": active_agents,
                "completed": completed_agents,
                "blocked": len(blocked_agents)
            },
            "integration_ready": integration_ready,
            "blocking_issues": [
                {"agent": a.name, "issues": a.blocking_issues}
                for a in blocked_agents if a.blocking_issues
            ],
            "next_actions": self._generate_next_actions()
        }
    
    def _generate_next_actions(self) -> List[str]:
        """Generate recommended next actions."""
        actions = []
        
        # Check for agents that can start
        for agent in self.agents.values():
            if agent.dependencies_met and agent.status in ["unknown", "idle"] and agent.deliverables_pending:
                actions.append(f"Start {agent.name} - {agent.deliverables_pending[0]}")
        
        # Check for integration opportunities
        if self.agents["agent_2"].progress_percentage > 50 and self.agents["agent_4"].progress_percentage > 30:
            actions.append("Begin dashboard-extractor integration testing")
        
        return actions[:5]  # Limit to top 5 actions
    
    def display_status_dashboard(self) -> None:
        """Display real-time status dashboard."""
        if not RICH_AVAILABLE:
            self._display_simple_status()
            return
        
        # Create layout
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=5)
        )
        
        layout["main"].split_row(
            Layout(name="agents"),
            Layout(name="progress")
        )
        
        def make_status_display():
            # Header
            layout["header"].update(Panel(
                f"🎯 SPARC Orchestrator - Agent Coordination Monitor\n"
                f"⏱️  Runtime: {datetime.now() - self.start_time}\n"
                f"📊 Overall Progress: {sum(a.progress_percentage for a in self.agents.values()) / len(self.agents):.1f}%",
                style="bold blue"
            ))
            
            # Agent status table
            table = Table(title="Agent Status")
            table.add_column("Agent", style="cyan", no_wrap=True)
            table.add_column("Status", style="magenta")
            table.add_column("Progress", style="green")
            table.add_column("Current Task", style="yellow")
            table.add_column("Files Modified", style="blue")
            
            for agent in self.agents.values():
                status_color = {
                    "active": "🟢 Active",
                    "idle": "🟡 Idle", 
                    "blocked": "🔴 Blocked",
                    "completed": "✅ Complete",
                    "unknown": "⚪ Unknown"
                }.get(agent.status, agent.status)
                
                files_modified = len(agent.files_modified)
                
                table.add_row(
                    agent.name[:30],
                    status_color,
                    f"{agent.progress_percentage:.1f}%",
                    agent.current_task[:20],
                    f"{files_modified} files"
                )
            
            layout["agents"].update(table)
            
            # Progress details
            progress_text = "📋 Deliverables Status:\n\n"
            for agent in self.agents.values():
                progress_text += f"🤖 {agent.name}:\n"
                progress_text += f"  ✅ Completed: {', '.join(agent.deliverables_completed) or 'None'}\n"
                progress_text += f"  🔄 Pending: {', '.join(agent.deliverables_pending[:2]) or 'None'}\n\n"
            
            layout["progress"].update(Panel(progress_text, title="Progress Details"))
            
            # Footer with next actions
            report = self.generate_status_report()
            footer_text = "🚀 Next Actions:\n" + "\n".join(f"• {action}" for action in report["next_actions"])
            layout["footer"].update(Panel(footer_text, style="bold green"))
            
            return layout
        
        with Live(make_status_display(), refresh_per_second=0.5) as live:
            try:
                while True:
                    time.sleep(self.monitoring_interval)
                    self.check_agent_progress()
                    live.update(make_status_display())
            except KeyboardInterrupt:
                console.print("\n[yellow]Monitoring stopped by user[/yellow]")
    
    def _display_simple_status(self) -> None:
        """Simple text-based status display when Rich is not available."""
        while True:
            try:
                os.system('clear' if os.name == 'posix' else 'cls')
                
                print("🎯 SPARC Orchestrator - Agent Coordination Monitor")
                print("=" * 60)
                print(f"⏱️  Runtime: {datetime.now() - self.start_time}")
                print(f"📊 Overall Progress: {sum(a.progress_percentage for a in self.agents.values()) / len(self.agents):.1f}%")
                print()
                
                print("Agent Status:")
                print("-" * 60)
                for agent in self.agents.values():
                    status_icon = {
                        "active": "🟢",
                        "idle": "🟡", 
                        "blocked": "🔴",
                        "completed": "✅",
                        "unknown": "⚪"
                    }.get(agent.status, "⚪")
                    
                    print(f"{status_icon} {agent.name[:40]:<40} {agent.progress_percentage:>6.1f}%")
                
                print()
                report = self.generate_status_report()
                print("🚀 Next Actions:")
                for action in report["next_actions"]:
                    print(f"  • {action}")
                
                time.sleep(self.monitoring_interval)
                self.check_agent_progress()
                
            except KeyboardInterrupt:
                print("\nMonitoring stopped by user")
                break
    
    def save_coordination_status(self) -> None:
        """Save current coordination status to file."""
        status_data = {
            "timestamp": datetime.now().isoformat(),
            "agents": {aid: asdict(agent) for aid, agent in self.agents.items()},
            "report": self.generate_status_report()
        }
        
        status_file = self.project_root / "coordination_status.json"
        with open(status_file, 'w') as f:
            json.dump(status_data, f, indent=2, default=str)


def main():
    """Main coordination monitor function."""
    print("🚀 Starting SPARC Orchestrator - Agent Coordination Monitor")
    
    monitor = AgentCoordinationMonitor()
    
    # Initial scan
    monitor.check_agent_progress()
    
    # Start monitoring
    try:
        monitor.display_status_dashboard()
    finally:
        monitor.save_coordination_status()
        print("✅ Coordination status saved")


if __name__ == "__main__":
    main()