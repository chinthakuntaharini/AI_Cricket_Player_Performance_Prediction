# Quick Setup: Add Your Semantic Scholar API Key

Your API key has been configured. To use it, create a `.env` file in the project root with the following content:

## Create .env file

Create a file named `.env` in the project root directory (`C:\Users\Harini\Downloads\R_A\.env`) with:

```
SEMANTIC_SCHOLAR_API_KEY=Lpesj1rrkxaP2zMWV0oqH2PNQN3KcoZR9tLNjmld
```

## Rate Limit Configuration

Your API key has a rate limit of **1 request per second (cumulative across all endpoints)**.

The system has been configured to:
- Use a 1.1 second delay between requests (slightly above the 1.0 second minimum)
- Track request timing to ensure we never exceed 1 RPS
- Automatically retry with exponential backoff if rate limit errors occur

## Verify It's Working

When you run the application, you should see:
```
Using Semantic Scholar API key (rate limit: 1 RPS cumulative across all endpoints)
```

## Important Notes

- **Never commit the `.env` file** - it's already in `.gitignore`
- The `.env` file contains your private API key - keep it secure
- Rate limit: 1 request per second (cumulative across all endpoints)
- The system will automatically respect this limit

## Test the Setup

Run:
```bash
python main.py "quantum machine learning"
```

The system will automatically use your API key and respect the 1 RPS rate limit.

