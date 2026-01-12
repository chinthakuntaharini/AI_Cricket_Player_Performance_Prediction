# Semantic Scholar API Key Setup Guide

## Why Get an API Key?

While the Semantic Scholar API works without authentication, using an API key provides:

- **Dedicated Rate Limits**: 1 request/second per endpoint (vs. shared 1000 RPS)
- **Better Reliability**: Less likely to be throttled during peak usage
- **Priority Support**: Semantic Scholar can better assist you if issues arise
- **Best Practice**: Recommended by Semantic Scholar for production use

## How to Get Your API Key

1. **Visit the API Key Request Form**:
   - Go to: https://www.semanticscholar.org/product/api#api-key-form
   - Fill out the form with your project details

2. **Wait for Email**:
   - You'll receive your private API key via email
   - **Important**: Do not share your API key with anyone

3. **Configure Your Project**:
   
   **Option A: Using .env file (Recommended)**
   - Create a `.env` file in the project root
   - Add your key:
     ```
     SEMANTIC_SCHOLAR_API_KEY=your_actual_api_key_here
     ```
   - The `.env` file is automatically loaded by the application

   **Option B: Environment Variable**
   - **Windows PowerShell**:
     ```powershell
     $env:SEMANTIC_SCHOLAR_API_KEY="your_actual_api_key_here"
     ```
   - **Windows Command Prompt**:
     ```cmd
     set SEMANTIC_SCHOLAR_API_KEY=your_actual_api_key_here
     ```
   - **Linux/Mac**:
     ```bash
     export SEMANTIC_SCHOLAR_API_KEY=your_actual_api_key_here
     ```

## Verify API Key is Working

When you run the application, you should see in the logs:
```
Using Semantic Scholar API key (rate limit: 1 RPS per endpoint)
```

If you see:
```
No API key provided - using unauthenticated access (shared rate limit)
```
Then the API key is not being loaded correctly.

## Security Notes

- Never commit your `.env` file or API key to version control
- The `.env` file is already in `.gitignore` (if using git)
- If your key is compromised, request a new one from Semantic Scholar

## Rate Limits

- **With API Key**: 1 request per second (1 RPS) per endpoint
- **Without API Key**: Shared 1000 requests/second (may be throttled)

The application automatically includes a 1.5-second delay between requests to respect these limits.

## References

- [Semantic Scholar API Overview](https://www.semanticscholar.org/product/api)
- [API Documentation](https://api.semanticscholar.org/api-docs/)
- [API Key Request Form](https://www.semanticscholar.org/product/api#api-key-form)

