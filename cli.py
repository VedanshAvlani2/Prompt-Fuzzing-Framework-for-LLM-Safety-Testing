import asyncio
from engine.orchestrator import Orchestrator

async def main():
    print("🚀 Starting attack engine orchestrator...")
    orch = Orchestrator()
    await orch.run_all()
    print("✅ Orchestrator finished execution.")

if __name__ == "__main__":
    asyncio.run(main())
