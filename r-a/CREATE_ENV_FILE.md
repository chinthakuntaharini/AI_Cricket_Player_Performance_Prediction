# Quick Guide: Create .env File for API Key

## Step 1: Create the .env file

Create a file named `.env` (exactly this name, with the dot at the beginning) in your project root:
```
C:\Users\Harini\Downloads\R_A\.env
```

## Step 2: Add your API key

Open the `.env` file in a text editor and add this line:

```
SEMANTIC_SCHOLAR_API_KEY=Lpesj1rrkxaP2zMWV0oqH2PNQN3KcoZR9tLNjmld
```

**Important**: 
- No spaces around the `=` sign
- No quotes around the key value
- Save the file

## Step 3: Verify it's working

Run this command to test:
```bash
python main.py "test query" --max-papers 1
```

You should see in the output:
```
Using Semantic Scholar API key (rate limit: 1 RPS cumulative across all endpoints)
```

If you see:
```
No API key provided - using unauthenticated access (shared rate limit)
```

Then the .env file isn't being loaded correctly.

## Rate Limit Configuration

Your API key has a rate limit of **1 request per second (cumulative across all endpoints)**.

The system is configured to:
- ✅ Use 1.1 second delay between requests (slightly above 1.0 second minimum)
- ✅ Track request timing to ensure we never exceed 1 RPS
- ✅ Automatically retry with exponential backoff if rate limit errors occur

## Windows PowerShell: Create .env file

You can create it using PowerShell:

```powershell
cd "C:\Users\Harini\Downloads\R_A"
@"
SEMANTIC_SCHOLAR_API_KEY=Lpesj1rrkxaP2zMWV0oqH2PNQN3KcoZR9tLNjmld
"@ | Out-File -FilePath .env -Encoding utf8
```

## Windows Command Prompt: Create .env file

```cmd
cd "C:\Users\Harini\Downloads\R_A"
echo SEMANTIC_SCHOLAR_API_KEY=Lpesj1rrkxaP2zMWV0oqH2PNQN3KcoZR9tLNjmld > .env
```

## Security Note

- The `.env` file is already in `.gitignore` - it won't be committed to git
- Never share your API key publicly
- If your key is compromised, request a new one from Semantic Scholar

