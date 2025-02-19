import uvicorn
from sequential_thinking.server import app
import os
from dotenv import load_dotenv

def main():
    # Load environment variables
    load_dotenv()
    
    # Get configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("DEBUG", "False").lower() == "true"
    
    # Run server
    uvicorn.run(
        "sequential_thinking.server:app",
        host=host,
        port=port,
        reload=debug
    )

if __name__ == "__main__":
    main()
