import uvicorn
from environment import index

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=index.IP,
        port=index.PORT,
        reload=True  # Enable auto-reload during development
    ) 