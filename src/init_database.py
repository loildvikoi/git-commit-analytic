def run():
    import asyncio
    from src.infrastructure.persistence.database import init_database
    asyncio.run(init_database())

if __name__  == "__main__":
    run()
