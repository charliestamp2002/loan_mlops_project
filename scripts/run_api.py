import uvicorn

def main() -> None:
    uvicorn.run(
        "loan_mlops.serving.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # nice for development
    )


if __name__ == "__main__":
    main()